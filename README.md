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

### Long-Term Atrial Fibrillation Database

Currently being integrated. Details will be updated as preprocessing is finalized.

---

## ⚙️ Preprocessing Pipeline (Planned)

- Resampling signals to a common frequency (128 Hz)
- Bandpass filtering (~0.5–40 Hz)
- Signal normalization
- Window segmentation (e.g., 10–30 seconds)
- Label alignment using .atr annotations

---

## 📊 Data Visualization

- ECG waveform visualization implemented for MIT-BIH AFDB
- Used for signal inspection, annotation verification, and debugging preprocessing steps

---

## 🚀 Future Work

- Complete integration of Long-Term AF Database
- Implement full preprocessing pipeline
- Train baseline model (CNN / RNN / Transformer) or find baseline references

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- PhysioNet for providing open-access ECG datasets
- Researchers behind the MIT-BIH and Long-Term AF databases
