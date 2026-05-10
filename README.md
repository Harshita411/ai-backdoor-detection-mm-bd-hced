# AI Backdoor Detection via MM-BD Reproduction and Hybrid Confidence–Entropy Analysis (HCED)

## 1. Introduction

This repository presents a structured reproduction of the work described in _“MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic” (IEEE S&P 2024)_, along with a **novel detection method proposed in this project**.

While MM-BD focuses on internal margin-based statistics for detecting backdoored models, this project introduces an original contribution:

**Hybrid Confidence–Entropy Detector (HCED)** — a lightweight, output-based detection mechanism designed and implemented as part of this work.

HCED is entirely **novel to this project** and provides an additional perspective for identifying anomalous model behavior using prediction distributions.

---

## 2. Objectives

### Reproduction

- Reproduce the MM-BD detection pipeline
- Validate implementation consistency with the original paper

### Novel Contribution

- Design and implement HCED (proposed in this project)
- Analyze prediction-level behavior of models
- Provide an additional detection signal complementary to MM-BD

---

## 3. Background: MM-BD

MM-BD detects backdoors by analyzing **maximum margin statistics** from intermediate representations of neural networks.

### Key Characteristics

- Works in a post-training setting
- Uses internal feature distributions
- Designed to detect arbitrary trigger patterns

---

## 4. Hybrid Confidence–Entropy Detector (HCED) — Proposed Method

### Contribution Statement

HCED is a **novel method introduced in this repository**. It is not part of the original MM-BD paper and represents the primary contribution of this work.

---

### Motivation

Neural networks encode significant behavioral information in their output probability distributions. HCED leverages this by capturing:

- Confidence separation between top predictions
- Prediction uncertainty

---

### Definition

Let:

- Δ(p) = Top-1 − Top-2 probability (confidence gap)
- H(p) = Entropy of prediction
- K = Number of classes

HCED(p) = Δ(p) × (1 - H(p) / log(K))

---

### Interpretation

- High confidence gap + low entropy → stronger model certainty
- Distribution patterns can be used to differentiate model behaviors

### Properties

- Model-agnostic
- No access to internal layers required
- Computationally lightweight
- Easily integrable with existing detection pipelines

---

## 5. Project Structure

MM-BD/
│
├── clean0/ # Clean trained models
├── model0/ # Backdoored models
├── src/ # Model architectures (ResNet)
│
├── run_clean.sh # Train clean models
├── run_attack.sh # Generate backdoored models
├── run_detect.sh # MM-BD detection
│
├── confidence_analysis.py # Behavioral analysis
├── hced.py # HCED implementation (novel work)
│
├── data/ # CIFAR-10 dataset
└── README.md

---

## 6. Experimental Setup

- **Dataset:** CIFAR-10
- **Model:** ResNet-18
- **Attack:** Standard backdoor injection
- **Metrics:** Detection analysis based on statistical signals

---

## 7. Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

### Install

pip install torch torchvision numpy matplotlib tqdm

---

## 8. Usage

### 1. Train Clean Models

./run_clean.sh

### 2. Train Backdoored Models

./run_attack.sh

### 3. Run MM-BD Detection

./run_detect.sh

---

### 4. Run HCED Detection

python hced.py

**Outputs:**

- HCED scores
- Clean vs Backdoored classification
- Histogram visualization (`hced_histogram.png`)

---

## 9. Results

| Model Type       | HCED Score | Classification |
| ---------------- | ---------- | -------------- |
| Clean Model      | ~0.70–0.73 | CLEAN          |
| Backdoored Model | ~0.78–0.82 | BACKDOORED     |

### Observations

- Clear statistical separation in HCED scores
- Efficient detection with minimal overhead

---

## 10. Supporting Analysis

Run:

python confidence_analysis.py

This demonstrates:

- Distinct confidence patterns across models
- Variation in entropy distributions

---

---

---

## 11. Conclusion

This project combines reproduction of an established backdoor detection method with a **newly proposed output-based detection approach (HCED)**. By incorporating both internal statistical analysis and external behavioral signals, it demonstrates a broader framework for analyzing model integrity in deep learning systems.

HCED represents a standalone, extensible contribution that can be applied across architectures without requiring access to internal model components.
