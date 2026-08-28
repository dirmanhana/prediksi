"""
train_lstm.py
=============
Model Deep Learning (LSTM) untuk membandingkan dengan XGBoost pada tugas yang
sama: prediksi arah pergerakan besok (naik/turun) — semua saham IDX.

Menggunakan data & split waktu yang SAMA dengan train_xgb.py agar adil:
  - Fitur dari data/features.db (SQLite)
  - Split 68% train / 12% valid / 20% test berdasarkan tanggal
  - Target: 1 jika close besok > close hari ini

Output:
  - data/model_lstm.pt
  - data/metrics_lstm.json
  - Perbandingan XGBoost vs LSTM dicetak di layar

Cara pakai:
  python train_lstm.py
"""

import json
import os
import sqlite3
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "features_macro.db")  # pakai fitur makro
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join(BASE, "data", "features.db")
OUT_MODEL = os.path.join(BASE, "data", "model_lstm.pt")
OUT_METRICS = os.path.join(BASE, "data", "metrics_lstm.json")
OUT_TEST_PROBS = os.path.join(BASE, "data", "lstm_test_probs.csv")
XGB_METRICS = os.path.join(BASE, "data", "metrics.json")

SEQ_LEN = 10          # jendela 10 hari terakhir
HIDDEN = 48           # unit LSTM
LAYERS = 2
BATCH = 1024
LR = 1e-3
EPOCHS = 15
PATIENCE = 3
RANDOM_STATE = 42
DEVICE = "cpu"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------- data
def load_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd_read_sql(conn)
    conn.close()
    df["tanggal"] = pd_to_datetime(df["tanggal"])
    df = df.sort_values(["ticker", "tanggal"]).reset_index(drop=True)
    return df


def pd_read_sql(conn):
    import pandas as pd
    table = "features_macro" if "macro" in DB_PATH else "features"
    return pd.read_sql(f"SELECT * FROM {table} ORDER BY ticker, tanggal", conn)


def pd_to_datetime(x):
    import pandas as pd
    return pd.to_datetime(x)


def make_windows(df, feat_cols, seq_len):
    """Buat (X, y, dates, tickers) — urut per saham, tanpa bocor antar saham."""
    import pandas as pd
    Xs, ys, dates, tks = [], [], [], []
    for tk, grp in df.groupby("ticker", sort=False):
        vals = grp[feat_cols].to_numpy(dtype=np.float32)
        tgt = grp["target"].to_numpy(dtype=np.float32)
        d = grp["tanggal"].to_numpy()
        n = len(grp)
        if n <= seq_len:
            continue
        for i in range(seq_len, n):
            Xs.append(vals[i - seq_len:i])
            ys.append(tgt[i])           # target = arah besok dari hari terakhir window
            dates.append(d[i])
            tks.append(tk)
    X = np.stack(Xs)                    # (N, seq_len, F)
    y = np.array(ys, dtype=np.float32)
    dates = pd.to_datetime(dates)
    return X, y, dates, tks


def split_by_date(X, y, dates):
    """Split berbasis waktu — cutoff tanggal sama dengan XGBoost."""
    d_np = dates.to_numpy().astype("datetime64[D]")
    uniq = np.sort(np.unique(d_np))
    n = len(uniq)
    cut_train = uniq[int(n * 0.68)]
    cut_valid = uniq[int(n * 0.80)]

    m_train = d_np <= cut_train
    m_valid = (d_np > cut_train) & (d_np <= cut_valid)
    m_test = d_np > cut_valid

    return (X[m_train], y[m_train]), (X[m_valid], y[m_valid]), (X[m_test], y[m_test])


# ---------------------------------------------------------------- model
class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=48, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------- train
def train_model(model, dl_train, dl_valid, epochs, patience):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, wait = -1, None, 0

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for xb, yb in dl_train:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        tr_loss = tot / n

        # validasi
        model.eval()
        vprobs = []
        with torch.no_grad():
            for xb, yb in dl_valid:
                vprobs.append(torch.sigmoid(model(xb)).numpy())
        vprobs = np.concatenate(vprobs)
        v_auc = roc_auc_score(dl_valid.dataset.ys, vprobs)
        print(f"  epoch {ep + 1:2d}/{epochs}  loss={tr_loss:.4f}  valid_auc={v_auc:.4f}  ({time.time() - t0:.0f}s)")

        if v_auc > best_auc:
            best_auc, wait = v_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stopping di epoch {ep + 1}")
                break

    model.load_state_dict(best_state)
    return best_auc


# ---------------------------------------------------------------- main
def main():
    import pandas as pd
    print("=" * 60)
    print("STEP 1: Load fitur dari SQLite & buat window LSTM")
    print("=" * 60)
    df = load_features()
    print(f"Fitur: {len(df):,} baris, {df['ticker'].nunique()} saham")

    meta = json.load(open(os.path.join(BASE, "data", "model_meta.pkl").replace(".pkl", ".json"))) if os.path.exists(
        os.path.join(BASE, "data", "model_meta.pkl").replace(".pkl", ".json")
    ) else None
    import joblib
    from train_xgb import FEATURE_COLS
    MACRO_COLS = [
        "ihsg_ret_1", "ihsg_ret_5", "ihsg_ret_20", "ihsg_sma20",
        "usd_ret_1", "usd_ret_5", "usd_ret_20", "usd_sma20",
    ]
    if "ihsg_ret_1" in df.columns:
        feat_cols = FEATURE_COLS + MACRO_COLS
    else:
        feat_cols = FEATURE_COLS
    print(f"Jumlah fitur: {len(feat_cols)}")

    X, y, dates, tickers = make_windows(df, feat_cols, SEQ_LEN)
    print(f"Sample LSTM: {X.shape[0]:,}  (jendela {SEQ_LEN} hari x {X.shape[2]} fitur)")

    # standardisasi fitur (fit di train saja, anti-leakage)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_by_date(X, y, dates)
    idx_tr, idx_va, idx_te = split_by_date(np.arange(len(X)), np.arange(len(X)), dates)
    _, _, te_idx = idx_tr, idx_va, idx_te
    scaler = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr = scaler.transform(Xtr.reshape(-1, Xtr.shape[2])).reshape(Xtr.shape)
    Xva = scaler.transform(Xva.reshape(-1, Xva.shape[2])).reshape(Xva.shape)
    Xte = scaler.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)
    print(f"Train: {len(Xtr):,} | Valid: {len(Xva):,} | Test: {len(Xte):,}")

    print("\n" + "=" * 60)
    print("STEP 2: Training LSTM (CPU)")
    print("=" * 60)

    class DS(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = torch.from_numpy(x)
            self.ys = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], torch.tensor(self.ys[i], dtype=torch.float32)

    dl_train = torch.utils.data.DataLoader(DS(Xtr, ytr), batch_size=BATCH, shuffle=True, num_workers=0)
    dl_valid = torch.utils.data.DataLoader(DS(Xva, yva), batch_size=BATCH, shuffle=False)

    model = LSTMNet(X.shape[2], hidden=HIDDEN, layers=LAYERS)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {n_params:,}")

    best_auc = train_model(model, dl_train, dl_valid, EPOCHS, PATIENCE)

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Evaluasi TEST")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        dl_test = torch.utils.data.DataLoader(DS(Xte, yte), batch_size=BATCH, shuffle=False)
        tprobs = np.concatenate([torch.sigmoid(model(xb)).numpy() for xb, _ in dl_test])

    # tuning threshold di validasi
    with torch.no_grad():
        vprobs = np.concatenate([torch.sigmoid(model(xb)).numpy() for xb, _ in dl_valid])
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(yva, (vprobs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    base = yte.mean()
    for label, th in [("threshold optimal (%.2f)" % best_t, best_t), ("threshold 0.50", 0.5)]:
        pred = (tprobs >= th).astype(int)
        print(f"\n--- TEST {label} ---")
        print(f"  Akurasi  : {accuracy_score(yte, pred):.4f}   (baseline mayoritas: {max(base, 1 - base):.4f})")
        print(f"  Precision: {precision_score(yte, pred, zero_division=0):.4f}")
        print(f"  Recall   : {recall_score(yte, pred, zero_division=0):.4f}")
        print(f"  F1       : {f1_score(yte, pred, zero_division=0):.4f}")
        print(f"  ROC-AUC  : {roc_auc_score(yte, tprobs):.4f}")

    torch.save({"model": model.state_dict(), "scaler": scaler,
                "feat_cols": feat_cols, "seq_len": SEQ_LEN}, OUT_MODEL)
    print(f"\nModel tersimpan: {OUT_MODEL}")

    # simpan proba test lengkap dgn ticker & tanggal utk ensemble
    te_probs_df = pd.DataFrame({
        "ticker": [tickers[i] for i in te_idx[0]],
        "tanggal": dates.to_numpy()[te_idx[0]],
        "prob": tprobs,
    })
    te_probs_df.to_csv(OUT_TEST_PROBS, index=False)
    print(f"Proba test LSTM: {OUT_TEST_PROBS} ({len(te_probs_df):,} baris)")

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERBANDINGAN XGBOOST vs LSTM (data TEST)")
    print("=" * 60)
    xgb_m = json.load(open(XGB_METRICS))

    def _metrics(name, m):
        return (name, m.get("roc_auc"), m.get("accuracy"), m.get("precision"),
                m.get("recall"), m.get("f1"), m.get("threshold"))

    lstm_m = {
        "roc_auc": round(float(roc_auc_score(yte, tprobs)), 4),
        "accuracy": round(float(accuracy_score(yte, (tprobs >= best_t).astype(int))), 4),
        "precision": round(float(precision_score(yte, (tprobs >= best_t).astype(int), zero_division=0)), 4),
        "recall": round(float(recall_score(yte, (tprobs >= best_t).astype(int), zero_division=0)), 4),
        "f1": round(float(f1_score(yte, (tprobs >= best_t).astype(int), zero_division=0)), 4),
        "threshold": round(best_t, 2),
    }
    row = [_metrics("XGBoost", xgb_m["test"]), _metrics("LSTM", lstm_m)]
    print(f"{'model':10s} {'AUC':>6s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'th':>5s}")
    for r in row:
        print(f"{r[0]:10s} {r[1]:6.3f} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f} {r[6]:5.2f}")

    with open(OUT_METRICS, "w") as f:
        json.dump({
            "model": "LSTM",
            "auc_test": round(float(roc_auc_score(yte, tprobs)), 4),
            "accuracy_test": round(float(accuracy_score(yte, (tprobs >= best_t).astype(int))), 4),
            "f1_test": round(float(f1_score(yte, (tprobs >= best_t).astype(int), zero_division=0)), 4),
            "optimal_threshold": round(best_t, 2),
            "best_valid_auc": round(float(best_auc), 4),
            "params": {"seq_len": SEQ_LEN, "hidden": HIDDEN, "layers": LAYERS,
                       "batch": BATCH, "lr": LR, "epochs": EPOCHS, "device": DEVICE},
            "xgboost_auc_test": xgb_m["test"]["roc_auc"],
        }, f, indent=2)
    print(f"Metrik LSTM: {OUT_METRICS}")


if __name__ == "__main__":
    main()
