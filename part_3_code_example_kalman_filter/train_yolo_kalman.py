# train_yolo_singleobj.py
import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt   # <-- added for loss plotting

# ------------- USER PARAMETERS -------------
CSV_SOURCE = "joined_filtered_labels_images_besttrack.csv"
FOLDER_PATH = r"images/TC_biendong_625x500-2015-2023_Himawari"
OUTPUT_RANDOM_GROUPS = "random_groups.csv"
KF_PRED_OUTPUT = "kf_predictions.csv"
IMG_WIDTH = 625
IMG_HEIGHT = 500
RNG_SEED = 42

COL_FILE = "file_name"
COL_X1 = "x1"
COL_Y1 = "y1"
COL_X2 = "x2"
COL_Y2 = "y2"

# candidate names for the international storm identifier in the CSV
INTL_ID_CANDIDATES = [
    "International_number_ID", "International_number", "international_number_id",
    "international_number", "intl_id", "International_number_ID", "int_num_id",
    "INTL_NUM", "International number ID"
]

# candidate names for timestamp columns or single datetime column
DATETIME_CANDIDATES = ["datetime", "date_time", "timestamp", "time"]
YEAR_CANDIDATES = ["year"]
MONTH_CANDIDATES = ["month"]
DAY_CANDIDATES = ["day"]
HOUR_CANDIDATES = ["hour"]
MIN_CANDIDATES = ["minute", "min"]

BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 20
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
GRID_S = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------


# ---------------- Utility functions ----------------
def ensure_long_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    # Windows long-path prefix handling
    if os.name == "nt":
        # ensure not already prefixed
        if not abs_path.startswith("\\\\?\\"):
            return "\\\\?\\" + abs_path
    return abs_path


def compute_center_from_bbox(row: pd.Series) -> Tuple[float, float]:
    x1, y1, x2, y2 = float(row[COL_X1]), float(row[COL_Y1]), float(row[COL_X2]), float(row[COL_Y2])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def iou_box(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = areaA + areaB - interArea
    if denom <= 0:
        return 0.0
    return float(interArea / denom)


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_datetime_from_row(row: pd.Series, df_cols: List[str]) -> Optional[datetime]:
    # prefer single datetime-like column
    for dt_col in DATETIME_CANDIDATES:
        if dt_col in df_cols:
            try:
                return pd.to_datetime(row[dt_col])
            except Exception:
                pass

    # fallback to components
    # year
    year_val = None
    for ycol in YEAR_CANDIDATES:
        if ycol in df_cols:
            try:
                year_val = int(row[ycol])
                break
            except Exception:
                year_val = None
    # month
    month_val = None
    for mcol in MONTH_CANDIDATES:
        if mcol in df_cols:
            try:
                month_val = int(row[mcol])
                break
            except Exception:
                month_val = None
    # day
    day_val = None
    for dcol in DAY_CANDIDATES:
        if dcol in df_cols:
            try:
                day_val = int(row[dcol])
                break
            except Exception:
                day_val = None
    # hour
    hour_val = 0
    for hcol in HOUR_CANDIDATES:
        if hcol in df_cols:
            try:
                hour_val = int(row[hcol])
                break
            except Exception:
                hour_val = 0
    # minute
    minute_val = 0
    for mincol in MIN_CANDIDATES:
        if mincol in df_cols:
            try:
                minute_val = int(row[mincol])
                break
            except Exception:
                minute_val = 0

    # require at least month and day or a year+month+day combination;
    # if missing year, default to 2000 to allow ordering and dt computation.
    if month_val is not None and day_val is not None:
        if year_val is None:
            year_val = 2000
        try:
            return datetime(year_val, month_val, day_val, hour_val, minute_val)
        except Exception:
            return None

    # if nothing parsable, return None
    return None


# ---------------- Kalman Filter (constant-velocity) ----------------
class KalmanFilterCV:
    """
    Simple 4D constant-velocity Kalman filter for tracking (x, y, vx, vy).
    State vector: [x, y, vx, vy]^T
    """

    def __init__(self, process_var: float = 1.0, measure_var: float = 10.0):
        # state estimate
        self.x = np.zeros((4, 1), dtype=float)
        # state covariance
        self.P = np.eye(4, dtype=float) * 500.0
        # process noise scale (will be multiplied into Q depending on dt)
        self.process_var = process_var
        # measurement covariance
        self.R = np.eye(2, dtype=float) * measure_var
        # measurement matrix
        self.H = np.zeros((2, 4), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4, dtype=float)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # continuous white noise approx for position & velocity
        q = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = np.zeros((4, 4), dtype=float)
        Q[0, 0] = dt4 / 4.0 * q
        Q[0, 2] = dt3 / 2.0 * q
        Q[1, 1] = dt4 / 4.0 * q
        Q[1, 3] = dt3 / 2.0 * q
        Q[2, 0] = dt3 / 2.0 * q
        Q[2, 2] = dt2 * q
        Q[3, 1] = dt3 / 2.0 * q
        Q[3, 3] = dt2 * q
        return Q

    def predict(self, dt: float = 1.0):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, meas: np.ndarray):
        # meas shape (2,) or (2,1)
        z = np.array(meas, dtype=float).reshape((2, 1))
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def set_state_from_measurement(self, meas: np.ndarray, next_meas: Optional[np.ndarray] = None, dt: float = 1.0):
        # initialize state from one or two measurements
        mx, my = float(meas[0]), float(meas[1])
        if next_meas is not None:
            nx, ny = float(next_meas[0]), float(next_meas[1])
            vx = (nx - mx) / dt
            vy = (ny - my) / dt
        else:
            vx, vy = 0.0, 0.0
        self.x = np.array([[mx], [my], [vx], [vy]], dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0


def apply_kalman_to_storm(df_storm: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Given a DataFrame for a single storm (with columns hurricane_center_x/hurricane_center_y and a parsed time_col),
    run a Kalman filter in time-order and produce filtered centers and next-step predictions.
    """
    if df_storm.shape[0] == 0:
        return df_storm

    # sort by time
    df_local = df_storm.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_local)
    # compute times in hours relative to first timestamp
    times = pd.to_datetime(df_local[time_col]).astype("datetime64[ns]")
    # convert to float hours
    t0 = times.iloc[0]
    hours = (times - t0) / np.timedelta64(1, "h")
    hours = hours.astype(float)
    dt_list = np.diff(hours, prepend=hours[0])

    # prepare output columns
    filtered_cx = []
    filtered_cy = []
    pred_next_cx = []
    pred_next_cy = []

    kf = KalmanFilterCV(process_var=1.0, measure_var=25.0)

    for i in range(n):
        dt = float(dt_list[i]) if dt_list[i] > 0 else 1.0
        cx = float(df_local.loc[i, "hurricane_center_x"])
        cy = float(df_local.loc[i, "hurricane_center_y"])

        if i == 0:
            # if at least 2 points exist, estimate initial velocity
            if n > 1:
                dt_init = float(max(1.0, float(dt_list[1])))
                nx = float(df_local.loc[1, "hurricane_center_x"])
                ny = float(df_local.loc[1, "hurricane_center_y"])
                kf.set_state_from_measurement(np.array([cx, cy]), np.array([nx, ny]), dt=dt_init)
            else:
                kf.set_state_from_measurement(np.array([cx, cy]), None, dt=1.0)
            # no predict step before first update
            kf.update(np.array([cx, cy]))
        else:
            # predict to current time
            kf.predict(dt)
            kf.update(np.array([cx, cy]))

        filtered_cx.append(float(kf.x[0, 0]))
        filtered_cy.append(float(kf.x[1, 0]))

        # predict to next timestamp (if available), otherwise predict +1 hour
        if i < n - 1:
            dt_to_next = float(max(1e-6, float(hours[i + 1] - hours[i])))
        else:
            dt_to_next = 1.0  # default prediction horizon of 1 hour

        # temporary copy of state for prediction
        x_backup = kf.x.copy()
        P_backup = kf.P.copy()

        kf.predict(dt_to_next)
        pred_next_cx.append(float(kf.x[0, 0]))
        pred_next_cy.append(float(kf.x[1, 0]))

        # restore state (so updates remain based on actual measurement sequence)
        kf.x = x_backup
        kf.P = P_backup

    df_local["kf_filtered_cx"] = filtered_cx
    df_local["kf_filtered_cy"] = filtered_cy
    df_local["kf_pred_next_cx"] = pred_next_cx
    df_local["kf_pred_next_cy"] = pred_next_cy
    return df_local


# ---------------- CSV load, split, save ----------------
def prepare_splits(csv_path: str,
                   folder_path: str,
                   out_csv: str,
                   seed: int = RNG_SEED,
                   frac_train=0.6,
                   frac_val=0.2) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    df_cols = list(df.columns)

    for c in (COL_FILE, COL_X1, COL_Y1, COL_X2, COL_Y2):
        if c not in df.columns:
            raise RuntimeError(f"CSV missing expected column: {c}")

    intl_col = find_column(df, INTL_ID_CANDIDATES)
    if intl_col is None:
        raise RuntimeError("No international storm identifier column found. Expected one of: " + ", ".join(INTL_ID_CANDIDATES))

    # compute centers and filter missing images
    valid_rows = []
    missing_count = 0
    folder_lp = ensure_long_path(folder_path)

    cx_list, cy_list = [], []
    intl_id_list = []
    time_list = []

    # attempt to locate/parse datetime for each row
    for _, row in df.iterrows():
        fname = str(row[COL_FILE]).strip()
        img_path = os.path.join(folder_lp, fname)
        if not os.path.exists(img_path):
            missing_count += 1
            continue
        cx, cy = compute_center_from_bbox(row)
        cx_list.append(cx)
        cy_list.append(cy)
        intl_id_list.append(row[intl_col])

        dt_obj = parse_datetime_from_row(row, df_cols)
        # if no timestamp found, attempt to infer from filename (best-effort)
        if dt_obj is None:
            # fallback: try to find yyyy-mm-dd or yyyymmdd or yyyy_mm_dd in filename
            try:
                fname_lower = fname.lower()
                found = None
                for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y_%m_%d"):
                    try:
                        start = None
                        # attempt to locate any 8 or 10 char date-like substring
                        for i in range(len(fname_lower) - len(fmt) + 1):
                            substr = fname_lower[i:i+len(fmt)]
                            try:
                                parsed = datetime.strptime(substr, fmt)
                                found = parsed
                                break
                            except Exception:
                                continue
                        if found is not None:
                            dt_obj = found
                            break
                    except Exception:
                        continue
            except Exception:
                dt_obj = None

        time_list.append(dt_obj)
        valid_rows.append(row)

    if missing_count > 0:
        print(f"WARNING: {missing_count} rows skipped due to missing image files.")

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    df_valid["hurricane_center_x"] = cx_list
    df_valid["hurricane_center_y"] = cy_list
    df_valid[intl_col] = intl_id_list
    df_valid["_parsed_time"] = time_list
    
    # ensure each storm group has parsed times; for rows missing times, a best-effort ordering will be used
    # apply Kalman filter per storm grouping
    grouped = []
    storms = []
    for storm_id, grp in df_valid.groupby(intl_col):
        storms.append(storm_id)
    
    kf_results = []
    for storm_id, grp in df_valid.groupby(intl_col):
        grp = grp.reset_index(drop=True)
    
        if grp["_parsed_time"].isnull().all():
            # create synthetic time ordering if necessary
            grp["_parsed_time"] = [datetime(2000, 1, 1) + pd.Timedelta(hours=i) for i in range(len(grp))]
        else:
            # where some timestamps missing, fill via interpolation
            grp["_parsed_time"] = pd.to_datetime(grp["_parsed_time"])
            if grp["_parsed_time"].isnull().any():
                # convert to numeric for interpolation
                grp["_parsed_time_num"] = grp["_parsed_time"].view(np.int64)
                grp["_parsed_time_num"] = grp["_parsed_time_num"].interpolate()
                grp["_parsed_time"] = pd.to_datetime(grp["_parsed_time_num"])
                grp.drop(columns="_parsed_time_num", inplace=True)
    
        df_kf = apply_kalman_to_storm(grp, "_parsed_time")
        kf_results.append(df_kf)
    
    if kf_results:
        df_kf_all = pd.concat(kf_results, ignore_index=True)
    else:
        df_kf_all = df_valid.copy()



    # merge Kalman outputs back onto df_valid by index-like keys (safe because original rows carried over)
    # ensure original row order preserved via file_name + bbox unique key
    # create a merge key
    df_valid["_merge_key"] = df_valid[COL_FILE].astype(str) + "|" + df_valid[COL_X1].astype(str) + "|" + df_valid[COL_Y1].astype(str)
    df_kf_all["_merge_key"] = df_kf_all[COL_FILE].astype(str) + "|" + df_kf_all[COL_X1].astype(str) + "|" + df_kf_all[COL_Y1].astype(str)
    df_merged = pd.merge(df_valid, df_kf_all[["_merge_key", "kf_filtered_cx", "kf_filtered_cy", "kf_pred_next_cx", "kf_pred_next_cy", "_parsed_time"]], on="_merge_key", how="left", suffixes=("", "_kf"))

    # save Kalman predictions
    df_merged.to_csv(KF_PRED_OUTPUT, index=False)
    print(f"Kalman filter predictions saved to {KF_PRED_OUTPUT}")

    # create train/val/test splits while keeping whole storms together
    rng = random.Random(seed)
    storm_ids = sorted(df_merged[intl_col].unique().tolist(), key=lambda x: str(x))
    rng.shuffle(storm_ids)
    n_storms = len(storm_ids)
    n_train = int(round(frac_train * n_storms))
    n_val = int(round(frac_val * n_storms))

    train_storms = set(storm_ids[:n_train])
    val_storms = set(storm_ids[n_train:n_train + n_val])
    test_storms = set(storm_ids[n_train + n_val:])

    def assign_group(sid):
        if sid in train_storms:
            return "train"
        elif sid in val_storms:
            return "val"
        else:
            return "test"

    df_merged["split_group"] = df_merged[intl_col].apply(assign_group)
    df_merged.to_csv(out_csv, index=False)
    print(f"Saved split CSV to {out_csv} with storm counts: total_storms={n_storms}, train={len(train_storms)}, val={len(val_storms)}, test={len(test_storms)}")
    return df_merged


# ---------------- Dataset ----------------
class SingleObjectYoloDataset(Dataset):
    def __init__(self, df: pd.DataFrame, folder_path: str, S: int = GRID_S,
                 img_size: Tuple[int, int] = IMG_SIZE, transform=None):
        self.df = df.reset_index(drop=True)
        self.folder_path = folder_path
        self.S = S
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.transform = transform
        self.folder_lp = ensure_long_path(folder_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row[COL_FILE]).strip()
        img_path = os.path.join(self.folder_lp, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_w, self.img_h))

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        x1, y1, x2, y2 = float(row[COL_X1]), float(row[COL_Y1]), float(row[COL_X2]), float(row[COL_Y2])

        scale_x = self.img_w / IMG_WIDTH
        scale_y = self.img_h / IMG_HEIGHT
        x1 *= scale_x; x2 *= scale_x; y1 *= scale_y; y2 *= scale_y

        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        cx = x1 + bbox_w / 2.0
        cy = y1 + bbox_h / 2.0

        cell_w = self.img_w / self.S
        cell_h = self.img_h / self.S
        cell_col = int(min(self.S - 1, math.floor(cx / cell_w)))
        cell_row = int(min(self.S - 1, math.floor(cy / cell_h)))

        bx = (cx - cell_col * cell_w) / cell_w
        by = (cy - cell_row * cell_h) / cell_h
        bw = bbox_w / self.img_w
        bh = bbox_h / self.img_h

        target = np.zeros((self.S, self.S, 5), dtype=np.float32)
        target[cell_row, cell_col, 0] = 1.0
        target[cell_row, cell_col, 1] = bx
        target[cell_row, cell_col, 2] = by
        target[cell_row, cell_col, 3] = bw
        target[cell_row, cell_col, 4] = bh

        return img_t, torch.from_numpy(target)


# ---------------- Model ----------------
class TinyYoloSingle(nn.Module):
    def __init__(self, S=GRID_S, pretrained=True):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=pretrained).features
        self.backbone = backbone
        self.adapt_pool = nn.AdaptiveAvgPool2d((S, S))
        self.head = nn.Conv2d(1280, 5, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.adapt_pool(feats)
        out = self.head(pooled)
        out = out.permute(0, 2, 3, 1)

        pc = torch.sigmoid(out[..., 0:1])
        box = torch.sigmoid(out[..., 1:])
        return torch.cat([pc, box], dim=-1)


# ---------------- Loss ----------------
def yolo_single_loss(pred, target, lambda_noobj=1.0, lambda_coord=5.0):
    pc_pred = pred[..., 0]
    box_pred = pred[..., 1:5]
    pc_t = target[..., 0]
    box_t = target[..., 1:5]

    obj_mask = (pc_t > 0.5).float()
    noobj_mask = 1.0 - obj_mask

    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    loss_obj = bce(pc_pred, pc_t)
    loss_obj_no = loss_obj * noobj_mask
    loss_obj_yes = loss_obj * obj_mask

    loss_box = mse(box_pred, box_t).sum(-1)

    return (
        lambda_coord * (loss_box * obj_mask).mean()
        + (loss_obj_yes.mean() + lambda_noobj * loss_obj_no.mean())
    )


# ---------------- Train / Eval ----------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss = yolo_single_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
    return running_loss / max(1, n)


def predict_boxes_from_output(out_tensor, S=GRID_S, img_w=IMG_WIDTH, img_h=IMG_HEIGHT):
    if out_tensor.ndim == 3:
        out = out_tensor
    else:
        out = out_tensor[0]

    pc = out[..., 0]
    box = out[..., 1:5]
    idx_flat = torch.argmax(pc.view(-1))
    r = int(idx_flat.item() // S)
    c = int(idx_flat.item() % S)
    bx, by, bw, bh = box[r, c].tolist()
    cell_w = img_w / S
    cell_h = img_h / S
    cx = (c + bx) * cell_w
    cy = (r + by) * cell_h
    w = bw * img_w
    h = bh * img_h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2], pc[r, c].item()


def evaluate_model(model, loader, device):
    model.eval()
    iou_list = []
    center_distances = []
    correct_005 = 0
    N = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            bs = imgs.size(0)
            for b in range(bs):
                pred_box, conf = predict_boxes_from_output(preds[b].cpu().float())
                target = targets[b].cpu().numpy()

                obj_indices = np.argwhere(target[..., 0] > 0.5)
                if obj_indices.shape[0] == 0:
                    continue
                r, c = obj_indices[0]
                bx, by, bw, bh = target[r, c, 1:5]

                cell_w = IMG_WIDTH / GRID_S
                cell_h = IMG_HEIGHT / GRID_S
                gt_cx = (c + bx) * cell_w
                gt_cy = (r + by) * cell_h
                gt_w = bw * IMG_WIDTH
                gt_h = bh * IMG_HEIGHT
                gt_box = [
                    gt_cx - gt_w / 2.0,
                    gt_cy - gt_h / 2.0,
                    gt_cx + gt_w / 2.0,
                    gt_cy + gt_h / 2.0,
                ]

                iou_val = iou_box(pred_box, gt_box)
                iou_list.append(iou_val)

                pred_cx = 0.5 * (pred_box[0] + pred_box[2])
                pred_cy = 0.5 * (pred_box[1] + pred_box[3])
                dist = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
                center_distances.append(dist)

                if iou_val >= 0.5:
                    correct_005 += 1
                N += 1

    avg_iou = float(np.mean(iou_list)) if iou_list else 0.0
    avg_dist = float(np.mean(center_distances)) if center_distances else 0.0
    recall_50 = correct_005 / max(1, N)
    return {"avg_iou": avg_iou, "avg_center_dist": avg_dist, "recall_iou50": recall_50, "n": N}


# ---------------- Main routine ----------------
def main(args):
    df_valid = prepare_splits(CSV_SOURCE, FOLDER_PATH, OUTPUT_RANDOM_GROUPS, seed=RNG_SEED)

    train_df = df_valid[df_valid["split_group"] == "train"].reset_index(drop=True)
    val_df = df_valid[df_valid["split_group"] == "val"].reset_index(drop=True)
    test_df = df_valid[df_valid["split_group"] == "test"].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    train_ds = SingleObjectYoloDataset(train_df, FOLDER_PATH, S=GRID_S, img_size=IMG_SIZE, transform=transform)
    val_ds = SingleObjectYoloDataset(val_df, FOLDER_PATH, S=GRID_S, img_size=IMG_SIZE, transform=transform)
    test_ds = SingleObjectYoloDataset(test_df, FOLDER_PATH, S=GRID_S, img_size=IMG_SIZE, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyYoloSingle(S=GRID_S, pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------------------------
    # Loss history containers
    # --------------------------
    train_losses = []
    val_losses = []   # validation loss computed using same loss function

    best_val_iou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)

        # compute validation loss directly
        model.eval()
        running_val = 0.0
        count_val = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(imgs)
                loss = yolo_single_loss(preds, targets)
                running_val += float(loss.item()) * imgs.size(0)
                count_val += imgs.size(0)
        val_loss = running_val / max(1, count_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        val_metrics = evaluate_model(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f} "
            f"| val_avg_iou={val_metrics['avg_iou']:.4f} "
            f"| val_center_dist={val_metrics['avg_center_dist']:.2f} px "
            f"| val_recall_iou50={val_metrics['recall_iou50']:.3f}"
        )

        if val_metrics["avg_iou"] > best_val_iou:
            best_val_iou = val_metrics["avg_iou"]
            ckpt_path = "best_tiny_yolo_single.pth"
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_iou": best_val_iou},
                       ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # --------------------------
    # Plot training / validation loss curve
    # --------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=180)
    print("Loss curve saved to loss_curve.png")

    # Final test evaluation
    if os.path.exists("best_tiny_yolo_single.pth"):
        ckpt = torch.load("best_tiny_yolo_single.pth", map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_model(model, test_loader, DEVICE)
    print("Final test metrics:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tiny single-object YOLO-like detector on CSV-labeled images with Kalman filter predictions per storm.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    NUM_EPOCHS = args.epochs
    main(args)
