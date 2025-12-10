# train_yolo_singleobj.py
import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt   # <-- added for loss plotting

# ------------- USER PARAMETERS -------------
CSV_SOURCE = "joined_filtered_labels_images_besttrack.csv"
FOLDER_PATH = r"images/TC_biendong_625x500-2015-2023_Himawari"
OUTPUT_RANDOM_GROUPS = "random_groups.csv"
IMG_WIDTH = 625
IMG_HEIGHT = 500
RNG_SEED = 42

COL_FILE = "file_name"
COL_X1 = "x1"
COL_Y1 = "y1"
COL_X2 = "x2"
COL_Y2 = "y2"

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
    if abs_path.startswith(r"\\?\\".replace("\\","\\")) or abs_path.startswith("\\\\?\\"):
        return abs_path
    if os.name == "nt":
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


# ---------------- CSV load, split, save ----------------
def prepare_splits(csv_path: str,
                   folder_path: str,
                   out_csv: str,
                   seed: int = RNG_SEED,
                   frac_train=0.6,
                   frac_val=0.2) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    for c in (COL_FILE, COL_X1, COL_Y1, COL_X2, COL_Y2):
        if c not in df.columns:
            raise RuntimeError(f"CSV missing expected column: {c}")

    #cx_list, cy_list = [], []
    valid_rows = []
    missing_count = 0
    folder_abs = os.path.abspath(folder_path)
    folder_lp = ensure_long_path(folder_path)

    cx_list, cy_list = [], []

    for _, row in df.iterrows():
        fname = str(row[COL_FILE]).strip()
        img_path = os.path.join(folder_lp, fname)
        if not os.path.exists(img_path):
            missing_count += 1
            continue
        cx, cy = compute_center_from_bbox(row)
        cx_list.append(cx)
        cy_list.append(cy)
        valid_rows.append(row)

    if missing_count > 0:
        print(f"WARNING: {missing_count} rows skipped due to missing image files.")

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    df_valid["hurricane_center_x"] = cx_list
    df_valid["hurricane_center_y"] = cy_list

    rng = random.Random(seed)
    indices = list(df_valid.index)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    group_col = []
    for i in df_valid.index:
        if i in train_idx:
            group_col.append("train")
        elif i in val_idx:
            group_col.append("val")
        else:
            group_col.append("test")

    df_valid["split_group"] = group_col
    df_valid.to_csv(out_csv, index=False)
    print(f"Saved split CSV to {out_csv} with counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return df_valid


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
    parser = argparse.ArgumentParser(description="Train tiny single-object YOLO-like detector on CSV-labeled images.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    NUM_EPOCHS = args.epochs
    main(args)
