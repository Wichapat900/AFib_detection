"""
extract_demo_segments.py

Extract:
1. AFib demo segment
2. Normal demo segment

from record 04043 and save them for Streamlit.
"""

import wfdb
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

RECORD = "04043"

AFDB_DIR = "MIT-BIH Atrial Fibrillation Database V1.0.0"
# AFDB_DIR = "data/ltaf"   # use this if that's your folder

FS = 128

# ============================================================
# AFIB SEGMENT
# ============================================================

AFIB_START = 1310.0
AFIB_END   = 1320.0

# ============================================================
# NORMAL SEGMENT
# ============================================================
# Replace these with a Normal region from your plot

NORMAL_START = 300.0
NORMAL_END   = 310.0

# ============================================================
# OUTPUT FOLDER
# ============================================================

Path("samples").mkdir(exist_ok=True)

# ============================================================
# LOAD RECORD
# ============================================================

record_path = str(Path(AFDB_DIR) / RECORD)

record = wfdb.rdrecord(record_path)

signal = record.p_signal[:, 0]  # Lead 0

# ============================================================
# EXTRACT AFIB
# ============================================================

afib_start_idx = int(AFIB_START * FS)
afib_end_idx   = int(AFIB_END * FS)

afib_segment = signal[afib_start_idx:afib_end_idx]

# ============================================================
# EXTRACT NORMAL
# ============================================================

normal_start_idx = int(NORMAL_START * FS)
normal_end_idx   = int(NORMAL_END * FS)

normal_segment = signal[normal_start_idx:normal_end_idx]

# ============================================================
# SAVE
# ============================================================

np.save("samples/afib_demo.npy", afib_segment)
np.save("samples/normal_demo.npy", normal_segment)

print("Saved:")
print(" samples/afib_demo.npy")
print(" samples/normal_demo.npy")

print()
print(f"AFib duration   : {len(afib_segment)/FS:.1f}s")
print(f"Normal duration : {len(normal_segment)/FS:.1f}s")