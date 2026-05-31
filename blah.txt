# 🫀 AFib Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Data-PhysioNet-orange)

A machine learning project focused on detecting **Atrial Fibrillation (AFib)** from long-term ECG recordings using publicly available datasets.

---

## 📌 Overview

This project aims to build an AFib detection model by combining and preprocessing multiple ECG datasets with different characteristics (sampling rates, durations, etc.).

Key challenges addressed:

- Merging datasets with **different sampling frequencies (128 Hz vs 250 Hz)**
- Handling **long-duration ECG recordings (10–25 hours)**
- Working with **noisy real-world signals**
- Simplifying labels into a **binary classification task (AFib vs Normal)**

---

## WEEK 4 progress

## 📂 Data Sources

This project uses two datasets from PhysioNet:

### 1. MIT-BIH Atrial Fibrillation Database

https://physionet.org/content/afdb/1.0.0/

### 2. Long-Term Atrial Fibrillation Database

https://physionet.org/content/ltafdb/1.0.0/

---

## 🧠 Dataset Details

### MIT-BIH Atrial Fibrillation Database

#### ✅ Records Used

04015, 04043, 04048, 04126, 04746, 04908, 04936,
05121, 05261, 06426, 06453, 06995, 07162, 07859,
07879, 07910, 08215, 08219, 08378, 08455

#### ❌ Excluded Records

00735, 03665, 08405, 08434, 08465, 08475, 05091

#### ⚠️ Minor problem (Used with Caution)

04043

#### 🏷️ Labels Used

| Class  | Labels           |
| ------ | ---------------- |
| AFib   | (AFIB, AFIB      |
| Normal | (N, N, (NSR, NSR |

#### 🚫 Ignored Labels

- Atrial Flutter: (AFL, AFL
- Junctional Rhythm: (J, J

---

## WEEK 5 progress

# Record Information & Data Visualization

## ✅ Records Used (23 Records)

04015, 04048, 04126, 04746, 04908, 04936, 05091,
05121, 05261, 06426, 06453, 06995, 07162, 07859,
07879, 07910, 08215, 08219, 08378, 08455, 08465,
08475

---

## ❌ Excluded Records

| Record | Reason                 |
| ------ | ---------------------- |
| 00735  | Signal unavailable     |
| 03665  | Signal unavailable     |
| 08405  | Unreadable data blocks |
| 08434  | Unreadable data blocks |
| 04043  | Bad blocks             |

> Note: Record `04043` contains partially corrupted blocks but remains usable for visualization/testing.

---

# Dataset File Explanation

| File   | Description                                                                            |
| ------ | -------------------------------------------------------------------------------------- |
| `.dat` | Raw ECG waveform data                                                                  |
| `.hea` | Metadata (record name, number of leads, sampling frequency, signal length, start time) |
| `.atr` | Rhythm annotations and labels (e.g. AFib, Normal rhythm)                               |
| `.qrs` | Detected R-peak locations                                                              |

---

# Planned Input

- Live ECG signal stream

---

# Planned Preprocessing Pipeline

- Bandpass filtering for noise reduction
- Resampling to **128 Hz**
- Signal normalization
- Window segmentation

---

# Planned Output

- AFib probability (%)
- SHAP-based explainability
- Explainable AI visualization

---

# Current Challenges

The Long-Term AF Database (LTAFDB) contains substantial noise and motion artifacts due to long-duration Holter monitor recordings.

---

# Future Training Plans

1. Patient-level data split
2. 80 / 10 / 10 train-validation-test split
3. 30-second windows with 50% overlap  
   (may reduce overlap to ~25%)
4. Compare multiple models:
   - XGBoost (XGB)
   - Random Forest (RF)
   - CNN
   - LSTM
   - CNN + LSTM hybrid
