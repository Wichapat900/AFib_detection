"""
preprocess_ltaf.py — Improved Long-Term AF Database Preprocessing
=================================================================

Changes from previous version:
- Fixed SEG_SAMPLES comment
- Bandpass BEFORE resampling
- Uses resample_poly instead of resample
- Cleaner pipeline structure
- Better numerical stability
- More deployment-consistent preprocessing

Outputs:
  - data/ltaf/signals.npy
  - data/ltaf/labels.npy
  - data/ltaf/patient_ids.npy
"""

import numpy as np
import wfdb
from pathlib import Path
from scipy.signal import butter, filtfilt, resample_poly
import json
import datetime
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DB_PATH = Path("Long Term AF Database V1.0.0")
OUT_PATH = Path("data/ltaf")

TARGET_FS = 128

SEG_LEN = 30
SEG_SAMPLES = SEG_LEN * TARGET_FS  # 3840

OVERLAP = 0.5

AFIB_LABELS = {"(AFIB", "AFIB"}
NORMAL_LABELS = {"(N", "N", "(NSR", "NSR"}

RECORDS = [
    "00","01","03","05","06","07","08","10",
    "100","101","102","103","104","105",
    "11","110","111","112","113","114","115","116","117","118","119",
    "12","120","121","122",
    "13","15","16","17","18","19","20",
    "200","201","202","203","204","205","206","207","208",
    "21","22","23","24","25","26","28",
    "30","32","33","34","35","37","38","39",
    "42","43","44","45","47","48","49",
    "51","53","54","55","56","58",
    "60","62","64","65","68","69","70","71","72","74","75",
]

# ─────────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────────

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = fs / 2

    highcut = min(highcut, nyq * 0.99)

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────────────────────
# LABEL MAPPING
# ─────────────────────────────────────────────────────────────

def get_rhythm_map(ann, signal_len):

    labels = np.full(signal_len, -1, dtype=np.int8)

    current_label = -1

    for i, sym in enumerate(ann.aux_note):

        sym_clean = sym.strip().rstrip("\x00")
        sample = ann.sample[i]

        if sym_clean in AFIB_LABELS:
            current_label = 1

        elif sym_clean in NORMAL_LABELS:
            current_label = 0

        elif sym_clean in ("(AFL", "AFL", "(J", "J"):
            current_label = -1

        if 0 <= sample < signal_len:
            labels[sample] = current_label

    # Forward-fill labels
    cur = -1

    for i in range(signal_len):

        if labels[i] != -1:
            cur = labels[i]

        labels[i] = cur

    return labels


# ─────────────────────────────────────────────────────────────
# SIGNAL CLEANING
# ─────────────────────────────────────────────────────────────

def fix_nan(signal):

    sig = signal.copy()

    nans = np.isnan(sig)

    if nans.any():
        idx = np.arange(len(sig))
        sig[nans] = np.interp(idx[nans], idx[~nans], sig[~nans])

    return sig


# ─────────────────────────────────────────────────────────────
# SEGMENT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_segments(signal, rhythm_map, fs, patient_id):

    step = int(SEG_SAMPLES * (1 - OVERLAP))

    segments = []
    labels = []
    patient_ids = []

    for start in range(0, len(signal) - int(SEG_LEN * fs), step):

        end_original = start + int(SEG_LEN * fs)

        seg = signal[start:end_original]
        seg_labels = rhythm_map[start:end_original]

        # Remove unknown labels
        known = seg_labels[seg_labels != -1]

        if len(known) < len(seg_labels) * 0.8:
            continue

        afib_frac = np.mean(known == 1)

        if afib_frac > 0.8:
            lbl = 1

        elif afib_frac < 0.2:
            lbl = 0

        else:
            continue

        # 1. FILTER FIRST
        seg = bandpass_filter(seg, fs)

        # 2. RESAMPLE SECOND
        if fs != TARGET_FS:
            seg = resample_poly(seg, TARGET_FS, int(fs))

        # Ensure exact length
        if len(seg) != SEG_SAMPLES:
            continue

        seg = seg.astype(np.float32)

        # 3. NORMALIZE
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)

        segments.append(seg)
        labels.append(lbl)
        patient_ids.append(patient_id)

    return segments, labels, patient_ids


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def preprocess_ltaf():

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    all_segments = []
    all_labels = []
    all_patient_ids = []

    skipped = []

    for patient_id, record_id in enumerate(RECORDS):

        rec_path = str(DB_PATH / record_id)

        if not (DB_PATH / f"{record_id}.dat").exists():

            print(f"[SKIP] {record_id} — file missing")
            skipped.append(record_id)
            continue

        try:
            rec = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, "atr")

        except Exception as e:

            print(f"[SKIP] {record_id} — {e}")
            skipped.append(record_id)
            continue

        signal = rec.p_signal[:, 0].astype(np.float32)

        signal = fix_nan(signal)

        fs = rec.fs

        rhythm_map = get_rhythm_map(ann, len(signal))

        segs, lbls, pids = extract_segments(
            signal,
            rhythm_map,
            fs,
            patient_id
        )

        print(
            f"{record_id} | "
            f"segments={len(segs)} | "
            f"AFib={sum(l == 1 for l in lbls)} | "
            f"Normal={sum(l == 0 for l in lbls)}"
        )

        all_segments.extend(segs)
        all_labels.extend(lbls)
        all_patient_ids.extend(pids)

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int8)
    g = np.array(all_patient_ids, dtype=np.int16)

    np.save(OUT_PATH / "signals.npy", X)
    np.save(OUT_PATH / "labels.npy", y)
    np.save(OUT_PATH / "patient_ids.npy", g)

    summary = {
        "total_segments": int(len(X)),
        "afib_segments": int(np.sum(y == 1)),
        "normal_segments": int(np.sum(y == 0)),
        "afib_pct": float(np.mean(y == 1) * 100),
        "normal_pct": float(np.mean(y == 0) * 100),
        "shape": list(X.shape),
        "sample_rate": TARGET_FS,
        "segment_length_sec": SEG_LEN,
        "skipped_records": skipped,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(OUT_PATH / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(f"Saved {len(X)} segments")
    print(f"Shape: {X.shape}")


if __name__ == "__main__":
    preprocess_ltaf()