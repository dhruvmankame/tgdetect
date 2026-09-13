"""Baselines used by the CTU-13 Prometheus paper-comparison suite.

These baselines intentionally live outside models/prometheus.py so the
paper-faithful Prometheus architecture remains untouched.

The base paper names several comparison systems but does not publish enough
implementation detail to reproduce their source code exactly.  Therefore:

* Static GNN / fine-tuned GNN reuse the repository's paper-aligned Prometheus
  GAT architecture and only change the training protocol.
* CNN-BiLSTM+attention is a transparent node-sequence baseline with two 1-D
  convolutions, a bidirectional LSTM whose concatenated width is 128, and
  additive attention.
* Generative replay uses a conditional VAE over CTU-13 node features plus a
  node MLP classifier.  It is explicitly a reproducible proxy for the paper's
  under-specified "generative replay IDS", not a claim about the authors'
  hidden implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMAttention(nn.Module):
    """Two Conv1d layers -> BiLSTM(128 total width) -> additive attention.

    The model predicts one class per node.  For a PyG mini-batch, each graph is
    treated as one variable-length node sequence and processed independently;
    logits are concatenated back in PyG node order.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 64,
        bilstm_total_hidden: int = 128,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if bilstm_total_hidden % 2:
            raise ValueError("bilstm_total_hidden must be even for bidirectional LSTM")
        self.in_channels = int(in_channels)
        self.conv_channels = int(conv_channels)
        self.bilstm_total_hidden = int(bilstm_total_hidden)
        self.dropout_p = float(dropout)
        per_direction = bilstm_total_hidden // 2

        self.conv1 = nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(bilstm_total_hidden, 1)
        self.classifier = nn.Linear(2 * bilstm_total_hidden, num_classes)

    def _one_graph(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, self.classifier.out_features))
        # [N,F] -> [1,F,N]
        h = x.transpose(0, 1).unsqueeze(0)
        h = self.dropout(F.relu(self.conv1(h)))
        h = self.dropout(F.relu(self.conv2(h)))
        # [1,C,N] -> [1,N,C]
        h = h.transpose(1, 2)
        seq, _ = self.bilstm(h)
        scores = self.attention(seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * seq, dim=1, keepdim=True)
        context = context.expand(-1, seq.size(1), -1)
        logits = self.classifier(torch.cat([seq, context], dim=-1))
        return logits.squeeze(0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            return self._one_graph(x)
        if batch.numel() == 0:
            return x.new_zeros((0, self.classifier.out_features))
        pieces = []
        num_graphs = int(batch.max().item()) + 1
        for graph_id in range(num_graphs):
            pieces.append(self._one_graph(x[batch == graph_id]))
        return torch.cat(pieces, dim=0) if pieces else x.new_zeros((0, 2))


class NodeMLP(nn.Module):
    """Simple node classifier used by the generative-replay proxy baseline."""

    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalVAE(nn.Module):
    """Conditional VAE over node features for a reproducible replay proxy."""

    def __init__(self, in_channels: int, latent_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_channels + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
        )

    @staticmethod
    def _one_hot(y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=2).float()

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, self._one_hot(y)], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, self._one_hot(y)], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        z = torch.randn(labels.numel(), self.latent_dim, device=labels.device)
        return self.decode(z, labels)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + float(beta) * kl

