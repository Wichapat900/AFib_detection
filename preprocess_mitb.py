"""
preprocess_afdb.py — Final Improved MIT-BIH AFDB Preprocessing
==============================================================

Features
--------
✓ Patient-wise dataset generation
✓ Sliding-window segmentation
✓ 50% overlap
✓ Clean AFib/Normal labeling
✓ Bandpass filtering
✓ Polyphase resampling (better ECG quality)
✓ NaN + corrupted block repair
✓ Z-score normalization
✓ Rhythm ambiguity exclusion
✓ Deployment-consistent preprocessing

Outputs
-------
data/mitb/signals.npy
data/mitb/labels.npy
data/mitb/patient_ids.npy
data/mitb/preprocess_summary.json
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

DB_PATH = Path("MIT-BIH Atrial Fibrillation Database V1.0.0")
OUT_PATH = Path("data/mitb")

TARGET_FS = 128

SEG_LEN = 30 # seconds
SEG_SAMPLES = SEG_LEN * TARGET_FS # 3840

OVERLAP = 0.5

# Keep exclusions minimal
EXCLUDE = {
    "00735",   # signals unavailable
    "03665",   # signals unavailable
    "08434",   # multiple unreadable blocks
    "08405",   # unreadable block
}

# Repairable corrupted records
BAD_BLOCKS = {
    "04043",
}

# All AFDB records
ALL_RECORDS = [
    "04015",
    "04043",
    "04048",
    "04126",
    "04746",
    "04908",
    "04936",
    "05091",
    "05121",
    "05261",
    "06426",
    "06453",
    "06995",
    "07162",
    "07859",
    "07879",
    "07910",
    "08215",
    "08219",
    "08378",
    "08455",
]

RECORDS = [r for r in ALL_RECORDS if r not in EXCLUDE]

AFIB_LABELS = {"(AFIB", "AFIB"}
NORMAL_LABELS = {"(N", "N", "(NSR", "NSR"}

EXCLUDED_RHYTHMS = {
    "(AFL", "AFL",
    "(J", "J"
}

# ─────────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────────

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):

    nyq = fs / 2

    # Prevent invalid cutoff
    highcut = min(highcut, nyq * 0.99)

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────────────────────
# SIGNAL REPAIR
# ─────────────────────────────────────────────────────────────

def repair_signal(signal):

    sig = signal.copy()

    # Fix NaNs
    nans = np.isnan(sig)

    if nans.any():

        idx = np.arange(len(sig))

        sig[nans] = np.interp(
            idx[nans],
            idx[~nans],
            sig[~nans]
        )

    # Fix long flat blocks
    i = 0

    while i < len(sig) - 1:

        if sig[i] == sig[i + 1]:

            j = i + 1

            while j < len(sig) and sig[j] == sig[i]:
                j += 1

            block_len = j - i

            if block_len > 500:

                if i > 0 and j < len(sig):

                    sig[i:j] = np.linspace(
                        sig[i - 1],
                        sig[j],
                        block_len
                    )

            i = j

        else:
            i += 1

    return sig


# ─────────────────────────────────────────────────────────────
# RHYTHM MAP
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

        elif sym_clean in EXCLUDED_RHYTHMS:

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
# SEGMENT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_segments(signal, rhythm_map, fs, patient_id):

    original_window = int(SEG_LEN * fs)

    step = int(original_window * (1 - OVERLAP))

    segments = []
    labels = []
    patient_ids = []

    for start in range(0, len(signal) - original_window, step):

        end = start + original_window

        seg = signal[start:end]
        seg_labels = rhythm_map[start:end]

        # Remove unknown regions
        known = seg_labels[seg_labels != -1]

        if len(known) < len(seg_labels) * 0.8:
            continue

        afib_frac = np.mean(known == 1)

        # High-confidence labels only
        if afib_frac > 0.8:
            lbl = 1

        elif afib_frac < 0.2:
            lbl = 0

        else:
            continue

        # ── 1. FILTER ─────────────────────────────
        seg = bandpass_filter(seg, fs)

        # ── 2. RESAMPLE ───────────────────────────
        if fs != TARGET_FS:

            seg = resample_poly(
                seg,
                TARGET_FS,
                int(fs)
            )

        # Ensure exact shape
        if len(seg) != SEG_SAMPLES:
            continue

        seg = seg.astype(np.float32)

        # ── 3. NORMALIZE ──────────────────────────
        seg = (
            seg - np.mean(seg)
        ) / (
            np.std(seg) + 1e-8
        )

        segments.append(seg)
        labels.append(lbl)
        patient_ids.append(patient_id)

    return segments, labels, patient_ids


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def preprocess():

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    all_segments = []
    all_labels = []
    all_patient_ids = []

    skipped = []

    print("\nStarting AFDB preprocessing...\n")

    for patient_id, record_id in enumerate(RECORDS):

        print(f"[{patient_id+1}/{len(RECORDS)}] {record_id}")

        rec_path = str(DB_PATH / record_id)

        try:

            record = wfdb.rdrecord(rec_path)

            ann = wfdb.rdann(rec_path, "atr")

        except Exception as e:

            print(f"  SKIPPED: {e}")

            skipped.append(record_id)

            continue

        signal = record.p_signal[:, 0].astype(np.float32)

        fs = record.fs

        # Repair corrupted records
        if (
            record_id in BAD_BLOCKS
            or np.isnan(signal).any()
        ):

            signal = repair_signal(signal)

        rhythm_map = get_rhythm_map(
            ann,
            len(signal)
        )

        segs, lbls, pids = extract_segments(
            signal,
            rhythm_map,
            fs,
            patient_id
        )

        print(
            f"  Segments={len(segs)} | "
            f"AFib={sum(l == 1 for l in lbls)} | "
            f"Normal={sum(l == 0 for l in lbls)}"
        )

        all_segments.extend(segs)
        all_labels.extend(lbls)
        all_patient_ids.extend(pids)

    # ─────────────────────────────────────────────
    # SAVE DATASETS
    # ─────────────────────────────────────────────

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int8)
    g = np.array(all_patient_ids, dtype=np.int16)

    np.save(OUT_PATH / "signals.npy", X)
    np.save(OUT_PATH / "labels.npy", y)
    np.save(OUT_PATH / "patient_ids.npy", g)

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────

    summary = {

        "total_segments":
            int(len(X)),

        "afib_segments":
            int(np.sum(y == 1)),

        "normal_segments":
            int(np.sum(y == 0)),

        "afib_pct":
            float(np.mean(y == 1) * 100),

        "normal_pct":
            float(np.mean(y == 0) * 100),

        "shape":
            list(X.shape),

        "sample_rate":
            TARGET_FS,

        "segment_length_sec":
            SEG_LEN,

        "overlap":
            OVERLAP,

        "excluded_records":
            list(EXCLUDE),

        "repaired_records":
            list(BAD_BLOCKS),

        "skipped_records":
            skipped,

        "timestamp":
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    with open(
        OUT_PATH / "preprocess_summary.json",
        "w"
    ) as f:

        json.dump(summary, f, indent=2)

    # ─────────────────────────────────────────────
    # FINAL STATS
    # ─────────────────────────────────────────────

    print("\n" + "=" * 60)

    print("PREPROCESSING COMPLETE")

    print("=" * 60)

    print(f"Total segments : {len(X)}")

    print(
        f"AFib segments  : {np.sum(y==1)} "
        f"({np.mean(y==1)*100:.1f}%)"
    )

    print(
        f"Normal segments: {np.sum(y==0)} "
        f"({np.mean(y==0)*100:.1f}%)"
    )

    print(f"Signal shape   : {X.shape}")

    print(f"Saved to       : {OUT_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    preprocess()