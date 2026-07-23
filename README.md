<div align="center">

<a href="Report.pdf">
<img src="dl3.png" width="100%" alt="Deep Learning for Finance">
</a>

# Deep Learning for Finance
### A Forensic Analysis of Overfitting, Data Leakage, and Walk-Forward Validation

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Investigating why seemingly profitable deep learning trading models often fail under proper evaluation.**

📄 **Click the banner above to read the full project report.**

</div>

---

# Overview

Financial machine learning papers frequently report predictive accuracies above random chance. Unfortunately, many of these gains disappear once common sources of **data leakage** and **backtest overfitting** are eliminated.

This project investigates that phenomenon using Bitcoin daily price data.

Three deep learning architectures are evaluated:

- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- CNN using Gramian Angular Fields (GAF)

Rather than focusing only on prediction performance, the project demonstrates how **look-ahead bias introduced through improper preprocessing** can create the illusion of profitable trading strategies.

---

# Research Question

> Can deep learning models genuinely predict future Bitcoin returns using historical prices alone, or are reported gains primarily caused by data leakage?

---

# Methodology

The project consists of three progressively more rigorous experiments.

## Step 1 — Static Train/Test Split (Leakage Introduced)

- Standard 80/20 split
- Global StandardScaler fitted on the entire dataset
- Deliberate look-ahead bias

Purpose:

Demonstrate how improper preprocessing can artificially improve performance.

---

## Step 2 — Walk-Forward Validation (Leakage Retained)

Rolling window evaluation using two configurations:

| Training | Testing |
|----------|---------|
|500 Days|500 Days|
|500 Days|100 Days|

The model is retrained as time progresses, but global scaling is intentionally retained.

Purpose:

Measure whether the apparent predictive edge survives a more realistic evaluation.

---

## Step 3 — Leakage-Free Pipeline

The preprocessing pipeline is corrected by:

- Splitting raw data first
- Fitting the scaler only on training data
- Transforming the test set using training statistics
- Generating GAF images inside each rolling window

Purpose:

Estimate the model's true predictive ability without look-ahead bias.

---

# Models

## Multi-Layer Perceptron (MLP)

Baseline dense neural network operating on rolling windows of historical returns.

```
Input
   ↓
Dense (64)
   ↓
Dropout
   ↓
Dense (32)
   ↓
Sigmoid
```

---

## Long Short-Term Memory (LSTM)

Designed to capture long-range temporal dependencies in financial time series.

---

## CNN + Gramian Angular Fields

Time series are converted into images using Gramian Angular Summation Fields (GASF), allowing convolutional neural networks to learn spatial representations of temporal behavior.

---

# Dataset

| Property | Value |
|-----------|-------|
|Asset|BTC-USD|
|Source|Yahoo Finance|
|Frequency|Daily|
|Period|2018–2024|
|Observations|≈2000|

---

# Results

| Experiment | MLP | Interpretation |
|------------|-----|---------------|
|Step 1 – Static Split + Leakage|**52.78%**|Artificial predictive edge|
|Step 2 – Walk Forward + Leakage|**52.86%**|Edge persists under retraining|
|Step 3 – Walk Forward (No Leakage)|**49.93%**|Returns to random chance|

## Key Finding

Removing data leakage eliminates the apparent predictive advantage.

The reported improvement from approximately **53% accuracy** to **50% accuracy** demonstrates that the earlier performance was driven primarily by **look-ahead bias**, not genuine market predictability.

---

# Experimental Pipeline

```
BTC-USD Data
      │
      ▼
 Feature Engineering
      │
      ▼
Rolling Windows
      │
      ▼
────────────────────────────────────────────
Step 1
Global Scaling
        ↓
 Train/Test Split
        ↓
 Models
────────────────────────────────────────────

────────────────────────────────────────────
Step 2
Global Scaling
        ↓
Walk Forward Validation
        ↓
 Models
────────────────────────────────────────────

────────────────────────────────────────────
Step 3
Split Raw Data
        ↓
Fit Scaler on Train Only
        ↓
Transform Test
        ↓
Walk Forward Validation
        ↓
 Models
────────────────────────────────────────────
```

---

# Repository Structure

```
.
├── main.ipynb              # Complete implementation
├── Report.pdf              # Technical report
├── TASKS.md                # Project milestones
├── requirements.txt
├── outputs/
│   ├── accuracy.png
│   ├── returns.png
│   ├── price_history.png
│   └── ...
├── LICENSE
└── README.md
```

---

# Installation

```bash
git clone https://github.com/sanaurrehmanarain/dl-finance-leakage.git

cd dl-finance-leakage

pip install -r requirements.txt
```

---

# Usage

Launch the notebook:

```bash
jupyter notebook main.ipynb
```

Run the notebook sequentially to reproduce every experiment.

---

# Key Takeaways

- Demonstrates how global normalization causes look-ahead bias.
- Implements leakage-free rolling walk-forward validation.
- Compares MLP, LSTM, and CNN-GAF architectures.
- Shows why backtests can overestimate predictive performance.
- Highlights the importance of robust validation in financial machine learning.

---

# References

- Wang, Z., & Oates, T. (2015). *Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks.*
- Yahoo Finance API
- TensorFlow
- Scikit-Learn

---

# Citation

If this repository contributes to your research, please cite:

```bibtex
@software{arain2026dlfinance,
  author = {Arain, Sana Ur Rehman},
  title = {Deep Learning for Finance: Analysis of Overfitting and Data Leakage},
  year = {2026},
  version = {1.0},
  url = {https://github.com/sanaurrehmanarain/dl-finance-leakage}
}
```

GitHub also provides citation formats automatically through the included `CITATION.cff` file.

---

# License

Released under the MIT License.

See the LICENSE file for details.
