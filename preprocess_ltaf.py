"""
preprocess_ltaf.py — Long Term AF Database Preprocessing
=========================================================
Prepares the MIT-BIH Long Term AF Database for cross-validation.

Outputs:
  - data/ltaf/signals.npy      (N, 7500) float32
  - data/ltaf/labels.npy       (N,) int  0=Normal 1=AFib
  - data/ltaf/patient_ids.npy  (N,) int

Usage:
  python src/preprocess_ltaf.py
"""

import numpy as np
import wfdb
from pathlib import Path
from scipy.signal import butter, filtfilt, resample
import json, datetime
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = Path("Long Term AF Database V1.0.0")
OUT_PATH    = Path("data/ltaf")
SAMPLE_RATE = 128
SEG_LEN     = 30
SEG_SAMPLES = SEG_LEN * SAMPLE_RATE  # 7500
OVERLAP     = 0.5

AFIB_LABELS   = {"(AFIB", "AFIB"}
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def bandpass(signal, fs, lo=0.5, hi=40.0):
    nyq = fs / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


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
    # Forward-fill
    cur = -1
    for i in range(signal_len):
        if labels[i] != -1:
            cur = labels[i]
        labels[i] = cur
    return labels


def extract_segments(signal, rhythm_map, fs, record_idx):
    step = int(SEG_SAMPLES * (1 - OVERLAP))
    segments, labels, pids = [], [], []
    for start in range(0, len(signal) - SEG_SAMPLES, step):
        end = start + SEG_SAMPLES
        seg_labels = rhythm_map[start:end]
        known = seg_labels[seg_labels != -1]
        if len(known) < SEG_SAMPLES * 0.8:
            continue
        afib_frac = np.mean(known == 1)
        if afib_frac > 0.8:
            lbl = 1
        elif afib_frac < 0.2:
            lbl = 0
        else:
            continue
        seg = signal[start:end].copy()
        if fs != SAMPLE_RATE:
            seg = resample(seg, SEG_SAMPLES).astype(np.float32)
        seg = bandpass(seg, SAMPLE_RATE).astype(np.float32)
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        segments.append(seg)
        labels.append(lbl)
        pids.append(record_idx)
    return segments, labels, pids


# ── Main ──────────────────────────────────────────────────────────────────────

def preprocess_ltaf():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    all_segs, all_labels, all_pids = [], [], []
    skipped = []

    for rec_idx, record_id in enumerate(RECORDS):
        rec_path = str(DB_PATH / record_id)

        # Check files exist
        if not (DB_PATH / f"{record_id}.dat").exists():
            print(f"  [{rec_idx+1}/{len(RECORDS)}] SKIP {record_id} — file not found")
            skipped.append(record_id)
            continue

        try:
            rec = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            print(f"  [{rec_idx+1}/{len(RECORDS)}] SKIP {record_id} — {e}")
            skipped.append(record_id)
            continue

        # Print what annotation symbols this record uses
        syms = set(s.strip().rstrip("\x00") for s in ann.aux_note if s.strip())
        fs   = rec.fs
        sig  = rec.p_signal[:, 0].astype(np.float32)

        # Fix NaN
        nans = np.isnan(sig)
        if nans.any():
            idx = np.arange(len(sig))
            sig[nans] = np.interp(idx[nans], idx[~nans], sig[~nans])

        rhythm_map = get_rhythm_map(ann, len(sig))
        segs, lbls, pids = extract_segments(sig, rhythm_map, fs, rec_idx)

        print(f"  [{rec_idx+1}/{len(RECORDS)}] {record_id} | fs={fs} | "
              f"segs={len(segs)} AFib={sum(l==1 for l in lbls)} "
              f"Normal={sum(l==0 for l in lbls)} | syms={syms}")

        all_segs.extend(segs)
        all_labels.extend(lbls)
        all_pids.extend(pids)

    X = np.array(all_segs,   dtype=np.float32)
    y = np.array(all_labels, dtype=np.int8)
    g = np.array(all_pids,   dtype=np.int16)

    np.save(OUT_PATH / "signals.npy",     X)
    np.save(OUT_PATH / "labels.npy",      y)
    np.save(OUT_PATH / "patient_ids.npy", g)

    print(f"\n{'='*50}")
    print(f"Saved {len(X)} segments")
    print(f"  AFib:   {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
    print(f"  Normal: {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
    print(f"  Shape:  {X.shape}")
    if skipped:
        print(f"  Skipped: {skipped}")
    print(f"Saved to {OUT_PATH}/")
    print(f"{'='*50}")

    summary = {
        "total_segments":  int(len(X)),
        "afib_segments":   int(np.sum(y==1)),
        "normal_segments": int(np.sum(y==0)),
        "afib_pct":        float(np.mean(y==1)*100),
        "normal_pct":      float(np.mean(y==0)*100),
        "shape":           list(X.shape),
        "skipped_records": skipped,
        "sample_rate":     SAMPLE_RATE,
        "seg_len_sec":     SEG_LEN,
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_PATH / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {OUT_PATH}/preprocess_summary.json")

if __name__ == "__main__":
    preprocess_ltaf()