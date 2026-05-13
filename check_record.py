import wfdb

path = "MIT-BIH Atrial Fibrillation Database V1.0.0/04936"
record = wfdb.rdrecord(path)

print(record.__dict__.keys())
print(record.sig_name)
print(record.p_signal.shape)

import wfdb

path = "MIT-BIH Atrial Fibrillation Database V1.0.0/04936"

ann = wfdb.rdann(path, 'atr')

print(ann.sample[:20])
print(ann.symbol[:20])