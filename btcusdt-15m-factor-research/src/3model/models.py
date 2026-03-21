from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

from .config import TARGET_COL, TRAIN_END
from .utils import time_split_mask


def train_xgb(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb(X_train, y_train, X_val, y_val):
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[early_stopping(50), log_evaluation(0)],
    )
    return model


class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last)


@dataclass
class EarlyStopper:
    patience: int = 8
    min_delta: float = 1e-5
    best_loss: float = float("inf")
    counter: int = 0
    best_state: dict = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_lstm_arrays(df_model, feature_cols, seq_len):
    full_idx = df_model.index
    scaler = StandardScaler()

    train_mask = time_split_mask(full_idx, full_idx.min(), TRAIN_END)
    scaler.fit(df_model.loc[train_mask, feature_cols])

    X_all = scaler.transform(df_model[feature_cols])
    y_all = df_model[TARGET_COL].to_numpy(dtype=np.float32)
    end_positions = np.arange(seq_len - 1, len(df_model))

    X_seq = np.stack([X_all[i - seq_len + 1 : i + 1] for i in end_positions])
    y_seq = y_all[end_positions]
    idx_seq = full_idx[end_positions]
    return X_seq, y_seq, idx_seq


def train_lstm(X_train, y_train, X_val, y_val, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=512, shuffle=False, drop_last=False)

    model = LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.20).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    stopper = EarlyStopper(patience=8, min_delta=1e-6)

    history = []
    for epoch in range(40):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print("[LSTM] epoch=%02d train_loss=%.8f val_loss=%.8f" % (epoch + 1, train_loss, val_loss))

        if stopper.step(val_loss, model):
            print("[LSTM] early stopping at epoch %d" % (epoch + 1))
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return model, pd.DataFrame(history)


def predict_lstm(model, X):
    device = next(model.parameters()).device
    loader = DataLoader(SeqDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=512, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).squeeze(-1).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)
