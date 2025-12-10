# train_dino_mlp.py
import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

# ------------------ DEBUG LOGGING ------------------
logging.basicConfig(filename="debug_output.log", level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------ USER PARAMETERS ------------------
CSV_SOURCE = "joined_filtered_labels_images_besttrack_filtered.csv"
FOLDER_PATH = r"images/TC_biendong_625x500-2015-2023_Himawari"
OUTPUT_RANDOM_GROUPS = "random_groups.csv"
IMG_WIDTH = 625
IMG_HEIGHT = 500
DOWNSAMPLE_FACTOR = 2  # reduces memory footprint
RNG_SEED = 42

COL_FILE = "file_name"
COL_X1 = "x1"
COL_Y1 = "y1"
COL_X2 = "x2"
COL_Y2 = "y2"

INTL_ID_CANDIDATES = [
    "International_number_ID", "International_number", "international_number_id",
    "international_number", "intl_id", "International_number_ID", "int_num_id",
    "INTL_NUM", "International number ID"
]

DATETIME_CANDIDATES = ["datetime", "date_time", "timestamp", "time"]
YEAR_CANDIDATES = ["year"]
MONTH_CANDIDATES = ["month"]
DAY_CANDIDATES = ["day"]
HOUR_CANDIDATES = ["hour"]
MIN_CANDIDATES = ["minute", "min"]

BATCH_SIZE = 4  # smaller batch to reduce memory
LR = 1e-4
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False

# ------------------ UTILITIES ------------------
def ensure_long_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        if not abs_path.startswith("\\\\?\\"):
            return "\\\\?\\" + abs_path
    return abs_path

def compute_center_from_bbox(row: pd.Series) -> Tuple[float, float]:
    x1, y1, x2, y2 = float(row[COL_X1]), float(row[COL_Y1]), float(row[COL_X2]), float(row[COL_Y2])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_datetime_from_row(row: pd.Series, df_cols: List[str]) -> Optional[datetime]:
    for dt_col in DATETIME_CANDIDATES:
        if dt_col in df_cols:
            try:
                return pd.to_datetime(row[dt_col])
            except Exception:
                continue
    year_val, month_val, day_val, hour_val, minute_val = None, None, None, 0, 0
    for ycol in YEAR_CANDIDATES:
        if ycol in df_cols:
            try: year_val = int(row[ycol]); break
            except Exception: continue
    for mcol in MONTH_CANDIDATES:
        if mcol in df_cols:
            try: month_val = int(row[mcol]); break
            except Exception: continue
    for dcol in DAY_CANDIDATES:
        if dcol in df_cols:
            try: day_val = int(row[dcol]); break
            except Exception: continue
    for hcol in HOUR_CANDIDATES:
        if hcol in df_cols:
            try: hour_val = int(row[hcol]); break
            except Exception: continue
    for mincol in MIN_CANDIDATES:
        if mincol in df_cols:
            try: minute_val = int(row[mincol]); break
            except Exception: continue
    if month_val is not None and day_val is not None:
        if year_val is None: year_val = 2000
        try: return datetime(year_val, month_val, day_val, hour_val, minute_val)
        except Exception: return None
    return None

# ---------------- CSV PREP ------------------
def prepare_splits(csv_path: str, folder_path: str, out_csv: str, seed: int=RNG_SEED,
                   frac_train=0.6, frac_val=0.2) -> pd.DataFrame:
    df = pd.read_csv(csv_path); df_cols = list(df.columns)
    logging.debug(f"Loaded CSV {csv_path} with {len(df)} rows")
    for c in (COL_FILE, COL_X1, COL_Y1, COL_X2, COL_Y2):
        if c not in df.columns: raise RuntimeError(f"CSV missing expected column: {c}")
    intl_col = find_column(df, INTL_ID_CANDIDATES)
    if intl_col is None: raise RuntimeError("No international storm identifier column found.")
    valid_rows = []; missing_count = 0; folder_lp = ensure_long_path(folder_path)
    cx_list, cy_list, intl_id_list, time_list = [], [], [], []
    for _, row in df.iterrows():
        fname = str(row[COL_FILE]).strip(); img_path = os.path.join(folder_lp, fname)
        if not os.path.exists(img_path): missing_count += 1; continue
        cx, cy = compute_center_from_bbox(row)
        cx_list.append(cx); cy_list.append(cy); intl_id_list.append(row[intl_col])
        dt_obj = parse_datetime_from_row(row, df_cols)
        time_list.append(dt_obj); valid_rows.append(row)
    if missing_count > 0: logging.warning(f"{missing_count} rows skipped due to missing image files.")
    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    df_valid["hurricane_center_x"] = cx_list; df_valid["hurricane_center_y"] = cy_list
    df_valid[intl_col] = intl_id_list; df_valid["_parsed_time"] = time_list

    rng = random.Random(seed)
    storm_ids = sorted(df_valid[intl_col].unique().tolist(), key=lambda x: str(x))
    rng.shuffle(storm_ids)
    n_storms = len(storm_ids); n_train = int(round(frac_train*n_storms)); n_val = int(round(frac_val*n_storms))
    train_storms = set(storm_ids[:n_train]); val_storms = set(storm_ids[n_train:n_train+n_val]); test_storms = set(storm_ids[n_train+n_val:])
    df_valid["split_group"] = df_valid[intl_col].apply(
        lambda sid: "train" if sid in train_storms else ("val" if sid in val_storms else "test")
    )
    df_valid.to_csv(out_csv, index=False)
    logging.info(f"Saved split CSV to {out_csv} with storm counts: total={n_storms}, train={len(train_storms)}, val={len(val_storms)}, test={len(test_storms)}")
    return df_valid

# ---------------- DATASET ------------------
class SingleObjectMLPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, folder_path: str, downsample: int = DOWNSAMPLE_FACTOR,
                 transform=None):
        self.df = df.reset_index(drop=True)
        self.folder_path = folder_path
        self.downsample = downsample
        self.transform = transform
        self.folder_lp = ensure_long_path(folder_path)
        self.img_w = IMG_WIDTH // downsample
        self.img_h = IMG_HEIGHT // downsample

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; fname = str(row[COL_FILE]).strip()
        img_path = os.path.join(self.folder_lp, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_w, self.img_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        # Flatten for MLP
        img_flat = img.view(-1)
        cx, cy = float(row[COL_X1]+row[COL_X2])/2.0, float(row[COL_Y1]+row[COL_Y2])/2.0
        target = torch.tensor([cx, cy], dtype=torch.float32)
        return img_flat, target

# ---------------- MODEL ------------------
class DINO_MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [2048, 512], output_dim: int = 2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ---------------- LOSS ------------------
def mse_loss(pred, target):
    return nn.MSELoss()(pred, target)

# ---------------- TRAIN / EVAL ------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train(); running_loss = 0.0; n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device); targets = targets.to(device)
        preds = model(imgs)
        loss = mse_loss(preds, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += float(loss.item())*imgs.size(0); n += imgs.size(0)
    return running_loss / max(1,n)

def evaluate_model(model, loader, device):
    model.eval(); running_loss = 0.0; n = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device); targets = targets.to(device)
            preds = model(imgs)
            loss = mse_loss(preds, targets)
            running_loss += float(loss.item())*imgs.size(0); n += imgs.size(0)
    return running_loss / max(1,n)

# ---------------- MAIN ------------------
def main(args):
    df_valid = prepare_splits(CSV_SOURCE, FOLDER_PATH, OUTPUT_RANDOM_GROUPS)
    train_df = df_valid[df_valid["split_group"]=="train"].reset_index(drop=True)
    val_df = df_valid[df_valid["split_group"]=="val"].reset_index(drop=True)
    test_df = df_valid[df_valid["split_group"]=="test"].reset_index(drop=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                                         std=[0.229,0.224,0.225])])
    train_ds = SingleObjectMLPDataset(train_df, FOLDER_PATH, downsample=DOWNSAMPLE_FACTOR, transform=transform)
    val_ds = SingleObjectMLPDataset(val_df, FOLDER_PATH, downsample=DOWNSAMPLE_FACTOR, transform=transform)
    test_ds = SingleObjectMLPDataset(test_df, FOLDER_PATH, downsample=DOWNSAMPLE_FACTOR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)

    input_dim = train_ds[0][0].numel()
    model = DINO_MLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate_model(model, val_loader, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # Evaluate on test set
    test_loss = evaluate_model(model, test_loader, DEVICE)
    logging.info(f"Final test MSE loss: {test_loss:.6f}")
    print(f"Final test MSE loss: {test_loss:.6f}")

    torch.save(model.state_dict(), "dino_mlp_model.pth")
    logging.info("Model saved to dino_mlp_model.pth")

    # ---------------- PLOT LOSS CURVES ------------------
    plt.figure(figsize=(8,5))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()
    logging.info("Training and validation loss curve saved to loss_curve.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train DINO MLP on hurricane CSV-labeled images")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    NUM_EPOCHS = args.epochs
    main(args)
