# 🛡️ AdaptiveShield

<div align="center">

### An End-to-End Adaptive Firewall Using Deep Q-Network Reinforcement Learning

*"Where others follow rules, we learn."*

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-SocketIO-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS%202017-purple?style=flat-square)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)
[![IEEE](https://img.shields.io/badge/Paper-IEEE%20NIGERCON%202026-red?style=flat-square)](https://github.com/Thinkersjnr1/ADAPTIVE-FIREWALL-USING-REINFORCEMENT-LEARNING)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

</div>

---

## 📌 Overview

**AdaptiveShield** is a production-deployable autonomous adaptive firewall powered by a **Deep Q-Network (DQN)** reinforcement learning agent. Unlike conventional static rule-based firewalls that can only block known threats, AdaptiveShield **continuously learns** optimal traffic filtering policies through real-time interaction with network traffic — no manual rule updates, no offline retraining required.

The system is trained and evaluated on the **CICIDS 2017** benchmark dataset — the most widely adopted standardised dataset in intrusion detection research — and deployed via a real-time **Python/Flask operational dashboard** with live packet classification, attack distribution visualisation, and DQN engine telemetry.

> Submitted to **IEEE NIGERCON 2026** — *AdaptiveShield: An End-to-End Adaptive Firewall Using Deep Q-Network Reinforcement Learning*

---

## ⚡ Performance

<div align="center">

| System | Accuracy | F1-Score | Detection Rate | Adaptability |
|--------|----------|----------|----------------|--------------|
| Rule-Based Firewall | ~60.0% | ~55.0% | Low | ❌ Static rules only |
| Random Forest (ML) | ~95.0% | ~94.0% | High | ❌ No online learning |
| **AdaptiveShield (DQN)** | **99.09%** | **99.49%** | **99.85%** | ✅ Continuous adaptation |

</div>

**AdaptiveShield outperforms the rule-based baseline by +39 percentage points and the Random Forest baseline by +4 percentage points** — while being the only system capable of autonomous online policy improvement without human intervention.

---

## 🧠 How It Works

AdaptiveShield formalises the firewall decision problem as a **Markov Decision Process (MDP)**:

```
State  (S) → 78-dimensional normalised CICIDS 2017 flow feature vector
Action (A) → {0: ALLOW,  1: BLOCK}
Reward (R) → Asymmetric outcome-based signal:
               +1.0  True Positive  (attack correctly blocked)
               +0.5  True Negative  (benign correctly allowed)
               -1.0  False Negative (attack missed)      ← heaviest penalty
               -0.5  False Positive (benign wrongly blocked)
```

The DQN agent learns the optimal policy **π\* = argmax E[Σ γᵗ · R(sₜ, aₜ)]** using:
- **Experience Replay** — 50,000-capacity replay buffer breaking temporal correlations
- **Target Network** — hard-updated every 500 steps for training stability
- **ε-greedy exploration** — decaying from 1.0 → 0.01 over 200,000 steps

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CICIDS 2017 Dataset                   │
│         ~2.8M flow records · 78 features · 15 classes  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Pre-processing Pipeline                    │
│  Concat → NaN fix → Impute → Encode → Split →          │
│  Normalise (MinMax) → SMOTE oversampling               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                DQN Neural Network                       │
│                                                         │
│   Input(78) → Dense(256,ReLU) → Dense(128,ReLU)       │
│             → Dense(64,ReLU)  → Output(2,Linear)       │
│                                                         │
│   Parameters: 61,506  │  Episodes: 500                 │
│   Discount γ: 0.95    │  Learning rate: 0.001          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│           Flask/SocketIO Operational Dashboard          │
│                                                         │
│  • Live packet classification feed                      │
│  • Real-time accuracy & reward timeline                 │
│  • Attack distribution heatmap                         │
│  • False negative tracking & alerts                    │
│  • System monitor (CPU, Memory, Threat Rate)           │
│  • Session summary with baseline comparison            │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AdaptiveShield/
│
├── dashboard.py              # Main Flask/SocketIO application & UI
│
├── src/
│   ├── preprocess.py         # 8-stage CICIDS 2017 preprocessing pipeline
│   ├── dqn_agent.py          # DQN agent (Q-network, replay buffer, training)
│   └── __init__.py
│
├── models/
│   └── dqn_firewall.keras    # Trained DQN model weights (add manually)
│
├── data/                     # Place CICIDS 2017 CSV files here (not tracked)
│
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Thinkersjnr1/ADAPTIVE-FIREWALL-USING-REINFORCEMENT-LEARNING.git
cd ADAPTIVE-FIREWALL-USING-REINFORCEMENT-LEARNING
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the **CICIDS 2017** dataset from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and place the CSV files inside the `data/` folder.

> ⚠️ The dataset is ~900MB and is excluded from this repository via `.gitignore`.

### 5. Add the trained model
Place your trained `dqn_firewall.keras` file inside the `models/` folder.

To train from scratch:
```bash
python src/train.py
```

### 6. Launch the dashboard
```bash
python dashboard.py
```

Open your browser at **http://localhost:5000**

> ℹ️ First startup takes 2–5 minutes while the CICIDS 2017 dataset is loaded and preprocessed. A staged loading screen shows progress.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Name | CICIDS 2017 (Canadian Institute for Cybersecurity) |
| Records | ~2.8 million labelled network flow records |
| Features | 78 numerical features per record |
| Classes | 15 (1 benign + 14 attack types) |
| Attack Types | DDoS, DoS Hulk, DoS GoldenEye, DoS Slowloris, PortScan, FTP-Patator, SSH-Patator, Web Attack (Brute Force, XSS, SQL Injection), Infiltration, Botnet, Heartbleed |
| Split | 80% train / 20% test (stratified) |
| Imbalance | Addressed via SMOTE on training set only |

---

## 🖥️ Dashboard Features

| Feature | Description |
|---------|-------------|
| 🎬 Staged Intro | Dramatic boot terminal → cinematic SVG shield logo → loading sequence |
| 📡 Live Feed | Real-time packet classification with decision, reward, and timestamp |
| 📈 Accuracy Timeline | DQN vs Rule-Based vs Random Forest comparison chart |
| 🎯 8 Metric Cards | Accuracy, Packets, Blocked, Allowed, FP, FN, F1, Precision — all expandable |
| 🔴 False Negative Tracking | Missed attacks highlighted in orange — invisible in most systems |
| 📊 Attack Distribution | Per-class breakdown showing both blocked and missed attacks |
| 🍩 Traffic Breakdown | 4-segment doughnut: TP / TN / FP / FN |
| ⚡ Reward History | Live bar chart of the last 40 DQN reward signals |
| 🖥️ System Monitor | Real-time CPU, Memory, and Threat Rate gauges |
| ⤢ Expand Mode | Every panel and metric card expands to full screen on click |
| 📋 Session Summary | Auto-opens on completion with full metrics and baseline comparison |
| ⬇️ Log Export | Download full session as CSV with all metrics per packet |
| 🔄 Restart | Reset and rerun analysis without refreshing the page |
| ⏸️ Pause / Resume | Pause packet processing mid-session |

---

## 🔧 DQN Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Episodes | 500 |
| Discount factor γ | 0.95 |
| Learning rate α | 0.001 (Adam) |
| Mini-batch size | 64 |
| Replay buffer | 50,000 |
| Target network update | Every 500 steps |
| ε start → min | 1.0 → 0.01 |
| ε decay steps | 200,000 (linear) |
| Gradient clip norm | 1.0 |

---

## 🔬 Confusion Matrix (Test Set)

```
                    Predicted ALLOW    Predicted BLOCK
Actual BENIGN            927 (TN)           73 (FP)
Actual ATTACK             13 (FN)         8,459 (TP)

Accuracy:        99.09%
Precision:       99.14%
Recall:          99.85%
F1-Score:        99.49%
False Pos. Rate:  7.30%
```

---

## 📦 Requirements

```
flask
flask-socketio
tensorflow>=2.0
numpy
scikit-learn
imbalanced-learn
psutil
eventlet
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.13, Flask, Flask-SocketIO |
| Machine Learning | TensorFlow/Keras, NumPy, scikit-learn |
| Data Balancing | imbalanced-learn (SMOTE) |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Fonts | Orbitron, Rajdhani, Share Tech Mono |
| Dataset | CICIDS 2017 |
| Model | Deep Q-Network (DQN) |

---

## 📄 Paper

If you use this work, please cite:

```bibtex
@inproceedings{yewenu2026adaptiveshield,
  title     = {AdaptiveShield: An End-to-End Adaptive Firewall Using Deep Q-Network Reinforcement Learning},
  author    = {Yewenu, Daniel and Okoya, Timileyin and Sofowora, Mayowa and Ogunsanwo, Olajide},
  booktitle = {Proceedings of IEEE NIGERCON 2026},
  year      = {2026},
  institution = {Lead City University, Ibadan, Nigeria}
}
```

---

## 🔮 Future Work

- [ ] Live packet capture integration (Scapy/tcpdump)
- [ ] Cloud-native / SDN deployment
- [ ] Adversarial RL for robustness testing
- [ ] Hybrid architectures (Transformer-DQN, CNN-DQN)
- [ ] Federated RL for distributed defence
- [ ] Explainable AI (XAI) for decision auditing
- [ ] ROC curve panel and CVSS severity mapping
- [ ] Docker containerisation
- [ ] Attack replay and what-if simulator

---

## 👤 Author

**Daniel Yewenu**
*B.Sc. Cybersecurity — Lead City University, Ibadan, Nigeria*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-daniel--yewenu-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/daniel-yewenu-45a370250)
[![GitHub](https://img.shields.io/badge/GitHub-Thinkersjnr1-black?style=flat-square&logo=github)](https://github.com/Thinkersjnr1)
[![Email](https://img.shields.io/badge/Email-yewenusewedo@gmail.com-red?style=flat-square&logo=gmail)](mailto:yewenusewedo@gmail.com)

---

## 📜 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**AdaptiveShield** — *Every attack makes it stronger.*

⭐ If this project helped you, please give it a star!

</div>
