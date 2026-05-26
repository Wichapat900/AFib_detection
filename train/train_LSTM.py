"""
train_lstm.py — Train AFibLSTM on LTAF, test on MITB
"""
from dataset import get_dataloaders
from model import AFibLSTM
from train.trainer import train

EPOCHS     = 30
BATCH_SIZE = 16
LR         = 1e-4
PATIENCE   = 10
SAVE_PATH  = "models/lstm_best.pth"

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

model = AFibLSTM()
train(model, train_loader, val_loader, SAVE_PATH, model_name="LSTM", epochs=EPOCHS, lr=LR, patience=PATIENCE)