"""
train_rnn.py -- Train AFibRNN on LTAF, test on MITB
"""
from dataset import get_dataloaders
from model import AFibRNN
from trainer import train

EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-3
PATIENCE   = 10
SAVE_PATH  = "models/rnn_best.pth"

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

model = AFibRNN()
train(model, train_loader, val_loader, SAVE_PATH, model_name="RNN", epochs=EPOCHS, lr=LR, patience=PATIENCE)