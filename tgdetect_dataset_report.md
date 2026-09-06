# TGDetect — Comprehensive Dataset Research Report
## AI-Based Multi-Step Cyberattack Detection Using Temporal Graph Neural Networks

*Compiled: 2026-08-03 · Sources: 5 local research papers (TAPAS, MAGIC, MGDA, IDS-HGAT, Huntrace), poster analysis, literature landscape, and deep web research across 7+ related papers*

---

## Table of Contents
1. [Datasets from Your 5 Research Papers](#part-1)
2. [Datasets from Related Papers (KAIROS, FLASH, NodLink, etc.)](#part-2)
3. [Datasets by Log Type (Network, Audit, Auth, DNS, Cloud)](#part-3)
4. [Additional Datasets from Deep Research](#part-4)
5. [Master Dataset Summary Table](#part-5)
6. [Recommended Dataset Strategy for TGDetect](#part-6)

---

<a id="part-1"></a>
## Part 1 — Datasets Extracted from Your 5 Research Papers

### 1.1 TAPAS (USENIX Security 2025) — Base Paper

> [!IMPORTANT]
> TAPAS is your base paper. It uses **2 major datasets** with **7 sub-datasets** total.

#### Dataset 1: DARPA TC Engagement 3 (E3)

| Property | Details |
|----------|---------|
| **Full Name** | DARPA Transparent Computing Engagement 3 |
| **Description** | Red-Blue confrontation exercise simulating enterprise APT attacks. Red team exploited vulnerabilities to exfiltrate data; blue teams collected kernel-level audit logs. |
| **Total Size** | **351.2 GB** raw audit logs |
| **Duration** | 2-week collection period, attacks across 17 hosts over 8 days |
| **Download Link** | 🔗 [github.com/darpa-i2o/Transparent-Computing](https://github.com/darpa-i2o/Transparent-Computing) |
| **Ground Truth** | 🔗 [Google Drive — Ground Truth Report](https://drive.google.com/open?id=1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK) |
| **Detector Labels** | 🔗 [github.com/threaTrace-detector/threaTrace](https://github.com/threaTrace-detector/threaTrace/) |
| **Source Code (TAPAS)** | 🔗 [doi.org/10.5281/zenodo.15610687](https://doi.org/10.5281/zenodo.15610687) |
| **Attack Labels** | ✅ Yes (ground-truth reports per engagement) |

**Sub-datasets used by TAPAS:**

| Sub-dataset | OS | Raw Size | TAPAS Storage | Storage Reduction | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **TC-Theia** | Ubuntu 12.04 | 79.3 GB | 81 MB | 1,002× | 98.54% | 96.01% | 97.32% |
| **CADETS** | FreeBSD 11.0 | 35.7 GB | 184 MB | 199× | 99.38% | 99.90% | 99.63% |
| **FiveDirections** | Windows 7 | 217.0 GB | 123 MB | 1,806× | 99.02% | 98.34% | 98.66% |
| **TRACE** | Ubuntu 14.04 | 19.2 GB | 208 MB | 95× | 99.45% | 100.00% | 99.72% |

#### Dataset 2: OpTC (Operationally Transparent Cyber)

| Property | Details |
|----------|---------|
| **Full Name** | Operationally Transparent Cyber (OpTC) |
| **Description** | 8-day enterprise-scale cyber exercise; 3 distinct attack campaigns over 3-day evaluation window |
| **Total Size** | **4.5 GB** |
| **Duration** | 8 days total, 3 hosts, avg ~5h attack duration |
| **Download Link** | 🔗 [Google Drive — OpTC Red Team Ground Truth](https://drive.google.com/drive/folders/1n3kkS3KR31KUegn42yk3e6JkZvf0Caa) |
| **Attack Labels** | ✅ Yes |

**OpTC Attack Scenarios in TAPAS:**

| Scenario | Attack Type | Precision | Recall | F1 |
|---|---|---|---|---|
| Attack 1 | PowerShell Empire | 97.29% | 96.82% | 97.05% |
| Attack 2 | Data Exfiltration | 97.83% | 96.24% | 96.97% |
| Attack 3 | Malware Upgrades | 97.91% | 96.71% | 97.29% |

---

### 1.2 MAGIC (USENIX Security 2024)

> MAGIC uses **3 dataset suites** totaling **131 GB** of audit logs.

#### Datasets Used:

| Dataset | Granularity | Size | Nodes | Edges | Download Link |
|---------|-------------|------|-------|-------|---------------|
| **DARPA E3-Trace** | Entity-level | 15.40 GB | 3,288,676 | 4,080,457 | 🔗 [github.com/darpa-i2o/Transparent-Computing](https://github.com/darpa-i2o/Transparent-Computing) |
| **DARPA E3-THEIA** | Entity-level | 17.91 GB | 1,623,966 | 2,874,821 | Same as above |
| **DARPA E3-CADETS** | Entity-level | 18.38 GB | 1,627,035 | 3,303,264 | Same as above |
| **StreamSpot** | Batch-level | 2.8 GB | 600 batches | — | 🔗 [github.com/sbustreamspot/sbustreamspot-data](https://github.com/sbustreamspot/sbustreamspot-data) |
| **Unicorn Wget** | Batch-level | 76.6 GB | 150 batches | — | 🔗 [Harvard Dataverse: unicorn-wget](https://dataverse.harvard.edu/dataverse/unicorn-wget) |

**StreamSpot Scenario Breakdown:**

| Scenario | Type | Batches | Avg Nodes | Avg Edges | Size |
|---|---|---|---|---|---|
| CNN | Benign | 100 | 8,989 | 294,903 | 0.9 GB |
| Download | Benign | 100 | 8,830 | 310,814 | 1.0 GB |
| Gmail | Benign | 100 | 6,826 | 37,382 | 0.1 GB |
| VGame | Benign | 100 | 8,636 | 112,958 | 0.4 GB |
| YouTube | Benign | 100 | 8,292 | 113,229 | 0.3 GB |
| **Attack** | Drive-by-Download | 100 | 8,890 | 28,423 | 0.1 GB |

**Unicorn Wget Breakdown:**

| Type | Batches | Avg Nodes | Avg Edges | Size |
|---|---|---|---|---|
| Benign | 125 | 265,424 | 975,226 | 64.0 GB |
| Attack (Supply-Chain) | 25 | 257,156 | 949,887 | 12.6 GB |

---

### 1.3 MGDA (2025)

> MGDA uses the **same 3 dataset suites** as MAGIC.

| Dataset | Precision | Recall | FPR | F1 |
|---------|-----------|--------|-----|-----|
| **StreamSpot** | 99.69% | 100.00% | 0.31% | 99.85% |
| **Unicorn Wget** | 96.11% | 93.00% | 3.96% | 94.46% |
| **DARPA E3-CADETS** | 94.32% | 99.77% | 0.22% | 96.97% |
| **DARPA E3-THEIA** | 98.50% | 99.99% | 0.12% | 99.24% |
| **DARPA E3-TRACE** | 99.26% | 99.98% | 0.08% | 99.62% |

Download links: Same as MAGIC (see §1.2 above).

---

### 1.4 IDS-HGAT (Computer Networks, Oct 2025)

> IDS-HGAT evaluates on **5 datasets** (based on paper abstract: "We evaluate IDS-HGAT on five datasets").

| Dataset | Notes |
|---------|-------|
| **DARPA TC E3 datasets** | CADETS, THEIA, TRACE (same as above) |
| **StreamSpot** | Same as above |
| **Unicorn Wget** | Same as above |

Performance highlights: 8% precision increase, 7.7% false alarm rate decrease vs. SOTA. 90.17–99.32% faster than UNICORN.

---

### 1.5 Huntrace (IEEE DSC 2025)

> Huntrace evaluates on **DARPA TC** provenance data.

| Property | Details |
|----------|---------|
| **Datasets** | DARPA Transparent Computing datasets (homogeneous provenance graphs) |
| **Download** | 🔗 [github.com/darpa-i2o/Transparent-Computing](https://github.com/darpa-i2o/Transparent-Computing) |
| **Performance** | F1 = 0.925, 1.7% F-score gain over UNICORN, 28.4% LOFPR reduction |
| **Scale** | 200K-node graphs, 10.5 GB memory, 142ms latency, 0.38 J/event |

---

<a id="part-2"></a>
## Part 2 — Datasets from Related Papers

### 2.1 KAIROS (IEEE S&P 2024)

| Property | Details |
|----------|---------|
| **Datasets** | DARPA TC E3 (CADETS/THEIA/TRACE), DARPA TC E5, OpTC |
| **Repository** | 🔗 [github.com/ubc-provenance/kairos](https://github.com/ubc-provenance/kairos) |
| **Notes** | Provides pre-trained models and processed data |

### 2.2 FLASH (IEEE S&P 2024)

| Property | Details |
|----------|---------|
| **Datasets** | DARPA TC E3, OpTC |
| **Notes** | Uses standard DARPA benchmarks; no custom dataset released |

### 2.3 NodLink (NDSS 2024)

| Property | Details |
|----------|---------|
| **Datasets** | Custom provenance dataset (Sangfor Technologies environment) |
| **Download** | 🔗 [github.com/PKU-ASAL/Simulated-Data](https://github.com/PKU-ASAL/Simulated-Data) |
| **Collection** | Ubuntu 20.04, Sysdig-based collection |
| **Notes** | Includes multi-stage APT attack chains with annotations |

### 2.4 ThreaTrace (IEEE TIFS 2022)

| Property | Details |
|----------|---------|
| **Datasets** | DARPA TC, StreamSpot, UNICORN |
| **Repository** | 🔗 [github.com/threaTrace-detector/threaTrace](https://github.com/threaTrace-detector/threaTrace/) |

### 2.5 UNICORN (NDSS 2020)

| Property | Details |
|----------|---------|
| **Datasets** | StreamSpot, DARPA TC, Unicorn Wget (custom supply-chain attack) |
| **Wget Dataset** | 🔗 [Harvard Dataverse: unicorn-wget](https://dataverse.harvard.edu/dataverse/unicorn-wget) |

### 2.6 ATLAS (USENIX Security 2021)

| Property | Details |
|----------|---------|
| **Datasets** | Custom dataset: 10 APT attack scenarios in virtual environment |
| **Log Types** | Windows Security Auditing, Firefox logs, DNS logs |
| **Download** | 🔗 [github.com/purseclab/ATLAS](https://github.com/purseclab/ATLAS) |
| **Notes** | ATLASv2 adds Sysmon and VMware Carbon Black Cloud logs |

### 2.7 HOLMES (IEEE S&P 2019)

| Property | Details |
|----------|---------|
| **Datasets** | Custom real-world APT scenarios based on Mandiant threat reports, auditd logs |
| **Notes** | No consolidated dataset release; modern comparisons use DARPA TC as proxy |

---

<a id="part-3"></a>
## Part 3 — Datasets by Log Type (From the Poster)

> [!IMPORTANT]
> The poster's right half (Problem Statement 3) identifies **5 log types** your project must support: **Audit Logs, Auth Logs, Network Logs, DNS Logs, and Cloud Logs**. Below are datasets for each.

### 3.1 🔒 Audit Logs (System-level audit trails)

| Dataset | Description | Size | Labels | Download Link |
|---------|-------------|------|--------|---------------|
| **DARPA TC E3** | System audit logs (Linux auditd, FreeBSD, Windows ETW) from APT exercises | 351.2 GB | ✅ | 🔗 [github.com/darpa-i2o/Transparent-Computing](https://github.com/darpa-i2o/Transparent-Computing) |
| **DARPA TC E5** | Larger-scale engagement with more hosts | Larger | ✅ | Same repo as above |
| **OpTC** | Enterprise-scale (~500 hosts) CAR-model audit events | Terabytes | ✅ | 🔗 [Google Drive](https://drive.google.com/drive/folders/1n3kkS3KR31KUegn42yk3e6JkZvf0Caa) / [Corrected DOI](https://doi.org/10.57745/UXCWOC) / [GitHub](https://github.com/FiveDirections/OpTC-data) |
| **LANL Unified Host & Network** | 90 days of host event logs from Los Alamos National Lab (12,425 users, 17,684 computers) | ~1 GB compressed | Partial (red team labels) | 🔗 [csr.lanl.gov/data/2017](https://csr.lanl.gov/data/2017/) |
| **StreamSpot** | SystemTap audit logs across 6 scenarios | 2.8 GB | ✅ | 🔗 [github.com/sbustreamspot/sbustreamspot-data](https://github.com/sbustreamspot/sbustreamspot-data) |
| **Unicorn Wget** | CamFlow kernel-level audit logs | 76.6 GB | ✅ | 🔗 [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/unicorn-wget) |
| **ATLAS Dataset** | Windows Security Auditing + Sysmon logs from 10 APT scenarios | — | ✅ | 🔗 [github.com/purseclab/ATLAS](https://github.com/purseclab/ATLAS) |

---

### 3.2 🔑 Auth Logs (Authentication & credential events)

| Dataset | Description | Size | Labels | Download Link |
|---------|-------------|------|--------|---------------|
| **LANL Auth Dataset** | 708 million authentication events over 58 consecutive days from LANL enterprise network (11,362 users, 22,284 computers) | ~1.2 GB compressed | ✅ (red team labels for user compromise) | 🔗 [csr.lanl.gov/data/cyber1](https://csr.lanl.gov/data/cyber1/) |
| **LANL Unified Host & Network** | Includes Windows auth events (Kerberos TGT/TGS, NTLM) across 90 days | ~1 GB compressed | Partial | 🔗 [csr.lanl.gov/data/2017](https://csr.lanl.gov/data/2017/) |
| **DARPA TC E3 (auth events)** | Login/auth events embedded in audit logs | Part of 351.2 GB | ✅ | 🔗 [github.com/darpa-i2o/Transparent-Computing](https://github.com/darpa-i2o/Transparent-Computing) |
| **Microsoft Authentication Dataset** | Azure AD/Entra ID sign-in logs | — | — | 🔗 Check [Microsoft Security Research datasets](https://www.microsoft.com/en-us/security/blog/) |
| **LANL Auth-Only Dataset** | 700M+ successful authentication events over 9 months (anonymized) | ~1-2 GB | Anomaly-based | 🔗 [csr.lanl.gov/data/auth](https://csr.lanl.gov/data/auth/) |

---

### 3.3 🌐 Network Logs (NetFlow, PCAP, firewall, IDS)

| Dataset | Description | Size | Labels | Download Link |
|---------|-------------|------|--------|---------------|
| **CICIDS 2017** | Network traffic with 80+ features, 14 attack types (DoS, DDoS, brute force, XSS, SQL injection, infiltration, port scan, botnet) | ~50 GB PCAP + CSV | ✅ | 🔗 [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **CSE-CIC-IDS 2018** | Updated version of CICIDS with 7 attack scenarios over 10 days | ~500 GB | ✅ | 🔗 [UNB Page](https://www.unb.ca/cic/datasets/ids-2018.html) / [AWS Registry](https://registry.opendata.aws/cse-cic-ids2018/) |
| **UNSW-NB15** | Network packets with 49 features, 9 attack families (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) | ~100 GB PCAP | ✅ | 🔗 [unsw.edu.au/unsw-nb15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| **CTU-13** | 13 botnet traffic capture scenarios from Czech Technical University | ~90 GB | ✅ | 🔗 [stratosphereips.org/datasets-ctu13](https://www.stratosphereips.org/datasets-ctu13) |
| **LANL Unified (NetFlow)** | 90 days of NetFlow from LANL's enterprise network | ~1 GB compressed | Partial | 🔗 [csr.lanl.gov/data/2017](https://csr.lanl.gov/data/2017/) |
| **ISCX 2012** | Network intrusion detection dataset | ~80 GB | ✅ | 🔗 [unb.ca/cic/datasets/ids.html](https://www.unb.ca/cic/datasets/ids.html) |
| **CIC-DDoS 2019** | DDoS-focused dataset with 13 attack types | Large | ✅ | 🔗 [unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html) |
| **MAWI Traffic Archive** | Real backbone traffic traces (daily 15-min samples) | Varies | ❌ (anomaly labels available) | 🔗 [mawi.wide.ad.jp/mawi](http://mawi.wide.ad.jp/mawi/) |
| **CAIDA Datasets** | Various network traffic datasets | Varies | Varies | 🔗 [caida.org/catalog](https://www.caida.org/catalog/) |

---

### 3.4 🔍 DNS Logs (DNS queries, responses, tunneling)

| Dataset | Description | Size | Labels | Download Link |
|---------|-------------|------|--------|---------------|
| **CIC-Bell-DNS-2021** | DNS over HTTPS (DoH) traffic with benign and malicious samples | — | ✅ | 🔗 [unb.ca/cic/datasets/dohbrw-2020.html](https://www.unb.ca/cic/datasets/dohbrw-2020.html) |
| **DNS Tunneling Datasets** | Labeled DNS tunneling traffic (dnscat2, iodine, dns2tcp) | ~100s MB | ✅ | 🔗 [github.com/ggyggy666/DNS-Tunnel-Datasets](https://github.com/ggyggy666/DNS-Tunnel-Datasets) |
| **DNS Malicious Domains** | ~90K domains (50% benign/malicious) with 34 features | ~10 MB | ✅ | 🔗 [Mendeley Data](https://doi.org/10.17632/623sshkdrz.5) |
| **CIRA-CIC-DoHBrw-2020** | DoH (DNS over HTTPS) browser traffic | — | ✅ | 🔗 [unb.ca/cic/datasets/dohbrw-2020.html](https://www.unb.ca/cic/datasets/dohbrw-2020.html) |
| **Majestic Million + DGA datasets** | Domain reputation + DGA domain lists | Varies | ✅ | 🔗 [majestic.com/reports/majestic-million](https://majestic.com/reports/majestic-million) |
| **DGTA Benchmark** | Domain Generation Algorithm detection benchmark | — | ✅ | 🔗 Search GitHub for "DGA detection dataset" |
| **Passive DNS datasets (Farsight/DNSDB)** | Passive DNS observation data | Large | ❌ | 🔗 [farsightsecurity.com](https://www.farsightsecurity.com/) (requires API key) |
| **ATLAS Dataset (DNS component)** | DNS logs from 10 APT attack scenarios | Part of ATLAS | ✅ | 🔗 [github.com/purseclab/ATLAS](https://github.com/purseclab/ATLAS) |
| **Alexa/Cisco Umbrella Top 1M** | Benign domain lists for baseline comparison | Small | ✅ (benign) | 🔗 [s3-us-west-1.amazonaws.com/umbrella-static/index.html](http://s3-us-west-1.amazonaws.com/umbrella-static/index.html) |

---

### 3.5 ☁️ Cloud Logs (AWS CloudTrail, Azure, cloud security events)

> [!WARNING]
> Cloud log datasets are the **scarcest** category. There are very few public labeled cloud security datasets available.

| Dataset | Description | Size | Labels | Download Link |
|---------|-------------|------|--------|---------------|
| **AWS CloudTrail Logs (FLAWS.cloud)** | 3.5 years of anonymized CloudTrail logs from vulnerable AWS setup | ~50 MB | Implicit | 🔗 [Kaggle](https://www.kaggle.com/datasets/aws-cloudtrails-dataset-flaws-cloud) / [flaws.cloud](http://flaws.cloud/) |
| **Invictus-IR AWS Dataset** | CloudTrail events from Stratus Red Team attack simulations, ATT&CK mapped | ~10-50 MB | ✅ (ATT&CK labeled) | 🔗 [github.com/invictus-ir/aws_dataset](https://github.com/invictus-ir/aws_dataset) |
| **Azure AD Sign-in Logs (Microsoft)** | Sample Azure AD/Entra ID authentication and activity logs | — | — | 🔗 [learn.microsoft.com/en-us/azure/active-directory/reports-monitoring](https://learn.microsoft.com/en-us/azure/active-directory/reports-monitoring/) |
| **EMBER (Elastic Malware Benchmark)** | While not cloud-native, used for cloud endpoint detection | — | ✅ | 🔗 [github.com/elastic/ember](https://github.com/elastic/ember) |
| **Splunk BOTS (Boss of the SOC)** | Multi-source log dataset including cloud, web, endpoint (simulated enterprise) | ~15 GB | ✅ | 🔗 [splunk.com/en_us/blog/security/boss-of-the-soc-scoring-server](https://bots.splunk.com/) |
| **Mordor / Security Datasets Project** | Pre-recorded security events (Windows, cloud services) mapped to ATT&CK | Varies | ✅ | 🔗 [github.com/OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets) |
| **Atomic Red Team Logs** | Generated logs from atomic tests across cloud and on-prem | Varies | ✅ | 🔗 [github.com/redcanaryco/atomic-red-team](https://github.com/redcanaryco/atomic-red-team) |
| **GCP Pub/Sub Audit Logs (Google)** | Sample GCP audit logs | — | — | Available via GCP trial |
| **Stratus Red Team** | Cloud attack simulation tool generating CloudTrail / Azure logs | — | ✅ (generated) | 🔗 [github.com/DataDog/stratus-red-team](https://github.com/DataDog/stratus-red-team) |

---

<a id="part-4"></a>
## Part 4 — Additional Datasets from Deep Research

### 4.1 Multi-Source / Comprehensive Cybersecurity Datasets

| Dataset | Description | Log Types | Download Link |
|---------|-------------|-----------|---------------|
| **LANL Cyber Security Dataset** | 58 days of auth, process, DNS, network from 12K+ users at Los Alamos National Lab | Auth + Process + DNS + Network | 🔗 [csr.lanl.gov/data/cyber1](https://csr.lanl.gov/data/cyber1/) |
| **LANL Unified Host & Network (2017)** | 90 days of host events + NetFlow from 17,684 computers | Host events + NetFlow + Auth | 🔗 [csr.lanl.gov/data/2017](https://csr.lanl.gov/data/2017/) |
| **Mordor / OTRF Security Datasets** | Pre-recorded events mapped to MITRE ATT&CK | Windows + Cloud + Network | 🔗 [github.com/OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets) |
| **Splunk BOTS v1/v2/v3** | SOC analyst competition data with multi-source logs | Web + Network + Endpoint + Cloud | 🔗 [bots.splunk.com](https://bots.splunk.com/) |
| **NodLink Simulated Data** | Ubuntu 20.04 provenance with multi-stage APT | System audit (Sysdig) | 🔗 [github.com/PKU-ASAL/Simulated-Data](https://github.com/PKU-ASAL/Simulated-Data) |
| **ATLAS v1/v2** | 10 APT scenarios with audit + DNS + browser logs | Audit + DNS + Browser | 🔗 [github.com/purseclab/ATLAS](https://github.com/purseclab/ATLAS) |

### 4.2 Recent Datasets & Frameworks (2024–2026)

| Name | Year | Description | Type |
|------|------|-------------|------|
| **ProvSyn** | 2026 | Framework to synthesize high-fidelity security provenance graphs for class-imbalance mitigation | Synthetic provenance |
| **S-DAPT-2026** | 2026 | Stage-aware synthetic dataset for multi-step APT detection evaluation | Synthetic APT |
| **CICAPT-IIoT** | 2025 | Provenance-based APT dataset tailored for Industrial IoT environments | IIoT provenance |
| **Windows-APT 2025** | 2025 | APT-inspired attack scenarios in Windows environments | Windows audit |
| **APT-KGL / APT-LMSPS** | 2025–26 | Heterogeneous provenance graph models with explicit node/edge typing | Typed provenance |

---

<a id="part-5"></a>
## Part 5 — Master Dataset Summary Table

> [!TIP]
> This table shows which datasets cover which log types relevant to your poster requirements.

| Dataset | Audit | Auth | Network | DNS | Cloud | Provenance Graph | APT Labels | Direct Link Available |
|---------|:-----:|:----:|:-------:|:---:|:-----:|:----------------:|:----------:|:--------------------:|
| **DARPA TC E3** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **DARPA TC E5** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **OpTC** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **StreamSpot** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Unicorn Wget** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **LANL Cyber1** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **LANL Unified 2017** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Partial | ✅ |
| **CICIDS 2017** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **CSE-CIC-IDS 2018** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **UNSW-NB15** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **CTU-13** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **ATLAS** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **CIC-DoHBrw-2020** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Mordor/OTRF** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Splunk BOTS** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Stratus Red Team** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **NodLink Data** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |

---

<a id="part-6"></a>
## Part 6 — Recommended Dataset Strategy for TGDetect

> [!IMPORTANT]
> Based on your project requirements (multi-step APT detection using temporal graph neural networks with audit, auth, network, DNS, and cloud logs), here is the recommended dataset strategy.

### Tier 1: Must-Have (Core Benchmarks — Used by All Your Papers)

These are **non-negotiable** for reproducibility and comparison against baselines:

| Priority | Dataset | Why | Link |
|----------|---------|-----|------|
| 🔴 P0 | **DARPA TC E3** (THEIA/TRACE/CADETS/FiveDirections) | Used by ALL 5 papers; gold standard for provenance-based APT detection | 🔗 [GitHub](https://github.com/darpa-i2o/Transparent-Computing) |
| 🔴 P0 | **OpTC** | Enterprise-scale validation; used by TAPAS, KAIROS, FLASH | 🔗 [Google Drive](https://drive.google.com/drive/folders/1n3kkS3KR31KUegn42yk3e6JkZvf0Caa) |
| 🟠 P1 | **StreamSpot** | Standard benchmark for batch-level detection; used by MAGIC, MGDA, IDS-HGAT | 🔗 [GitHub](https://github.com/sbustreamspot/sbustreamspot-data) |
| 🟠 P1 | **Unicorn Wget** | Hardest benchmark (stealth supply-chain); used by MAGIC, MGDA, UNICORN | 🔗 [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/unicorn-wget) |

### Tier 2: Should-Have (Multi-Source Log Coverage)

These fill the **auth, DNS, and network log gaps** not covered by DARPA TC:

| Priority | Dataset | Fills Gap | Link |
|----------|---------|-----------|------|
| 🟡 P2 | **LANL Cyber1** | Auth + DNS + Network | 🔗 [LANL](https://csr.lanl.gov/data/cyber1/) |
| 🟡 P2 | **LANL Unified 2017** | Host + Auth + NetFlow + DNS (all in one!) | 🔗 [LANL](https://csr.lanl.gov/data/2017/) |
| 🟡 P2 | **CICIDS 2017** | Network IDS logs | 🔗 [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |
| 🟡 P2 | **ATLAS** | Audit + DNS from APT scenarios | 🔗 [GitHub](https://github.com/purseclab/ATLAS) |

### Tier 3: Nice-to-Have (Cloud & Extended Coverage)

These address the **cloud log gap** and provide additional diversity:

| Priority | Dataset | Fills Gap | Link |
|----------|---------|-----------|------|
| 🟢 P3 | **Mordor/OTRF Security Datasets** | Multi-source including cloud, ATT&CK mapped | 🔗 [GitHub](https://github.com/OTRF/Security-Datasets) |
| 🟢 P3 | **Splunk BOTS** | Cloud + endpoint + network (realistic SOC scenario) | 🔗 [Splunk](https://bots.splunk.com/) |
| 🟢 P3 | **Stratus Red Team** | Cloud attack simulation (CloudTrail/Azure) | 🔗 [GitHub](https://github.com/DataDog/stratus-red-team) |
| 🟢 P3 | **CIC-DoHBrw-2020** | DNS over HTTPS detection | 🔗 [UNB](https://www.unb.ca/cic/datasets/dohbrw-2020.html) |

### Coverage Check Against Poster Requirements

| Log Type | Covered By |
|----------|------------|
| ✅ **Audit Logs** | DARPA TC E3/E5, OpTC, StreamSpot, Unicorn Wget, LANL, ATLAS |
| ✅ **Auth Logs** | LANL Cyber1, LANL Unified 2017, DARPA TC (embedded) |
| ✅ **Network Logs** | CICIDS 2017/2018, UNSW-NB15, CTU-13, LANL Unified 2017 |
| ✅ **DNS Logs** | LANL Cyber1, ATLAS, CIC-DoHBrw-2020 |
| ⚠️ **Cloud Logs** | Mordor/OTRF, Splunk BOTS, Stratus Red Team (limited availability) |

> [!NOTE]
> Cloud log datasets remain the weakest area in public cybersecurity research. Consider generating your own using **Stratus Red Team** (for AWS CloudTrail) or **Atomic Red Team** (for Azure) to fill this gap.

---

## Quick-Access Download Links (All in One Place)

```
# Core Provenance Datasets
DARPA TC:        https://github.com/darpa-i2o/Transparent-Computing
OpTC:            https://drive.google.com/drive/folders/1n3kkS3KR31KUegn42yk3e6JkZvf0Caa
StreamSpot:      https://github.com/sbustreamspot/sbustreamspot-data
Unicorn Wget:    https://dataverse.harvard.edu/dataverse/unicorn-wget
KAIROS repo:     https://github.com/ubc-provenance/kairos
ThreaTrace:      https://github.com/threaTrace-detector/threaTrace/
NodLink:         https://github.com/PKU-ASAL/Simulated-Data
ATLAS:           https://github.com/purseclab/ATLAS
TAPAS code:      https://doi.org/10.5281/zenodo.15610687

# Network/IDS Datasets
CICIDS 2017:     https://www.unb.ca/cic/datasets/ids-2017.html
CIC-IDS 2018:    https://registry.opendata.aws/cse-cic-ids2018/
UNSW-NB15:       https://research.unsw.edu.au/projects/unsw-nb15-dataset
CTU-13:          https://www.stratosphereips.org/datasets-ctu13

# Auth/DNS/Multi-Source Datasets
LANL Cyber1:     https://csr.lanl.gov/data/cyber1/
LANL Unified:    https://csr.lanl.gov/data/2017/
CIC-DoHBrw:      https://www.unb.ca/cic/datasets/dohbrw-2020.html

# Cloud / SOC Datasets
Mordor/OTRF:     https://github.com/OTRF/Security-Datasets
Splunk BOTS:     https://bots.splunk.com/
Stratus Red Team: https://github.com/DataDog/stratus-red-team
Atomic Red Team: https://github.com/redcanaryco/atomic-red-team

# Cloud Datasets
Invictus-IR AWS: https://github.com/invictus-ir/aws_dataset
flAWS CloudTrail: https://www.kaggle.com/datasets/aws-cloudtrails-dataset-flaws-cloud

# DNS Datasets
DNS Tunneling:   https://github.com/ggyggy666/DNS-Tunnel-Datasets
DNS Mal Domains: https://doi.org/10.17632/623sshkdrz.5

# Ground Truth
DARPA GT Report: https://drive.google.com/open?id=1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK
OpTC Corrected:  https://doi.org/10.57745/UXCWOC
LANL Auth-Only:  https://csr.lanl.gov/data/auth/
```

---

*This report consolidates findings from 5 local research papers (TAPAS, MAGIC, MGDA, IDS-HGAT, Huntrace), 7 related papers (KAIROS, FLASH, NodLink, ThreaTrace, UNICORN, ATLAS, HOLMES), poster analysis, literature landscape documents, and deep web research.*
