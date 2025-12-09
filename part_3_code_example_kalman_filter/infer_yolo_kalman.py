import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
GRID_S = 7
IMG_SIZE = 448
IMG_WIDTH = 625
IMG_HEIGHT = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "random_groups.csv"
image_folder = "images/TC_biendong_625x500-2015-2023_Himawari"
weights_path = "best_tiny_yolo_single.pth"
output_examples = "pred_examples/"
os.makedirs(output_examples, exist_ok=True)

intl_col = "International number ID"  # the storm ID column

# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Utility: Grid -> Pixel Box
# ---------------------------------------------------------
def grid_to_bbox(pred_cell, cell_i, cell_j, S, img_size=IMG_SIZE):
    bx, by, bw, bh = pred_cell[1:]
    cell_w = img_size / S
    cell_h = img_size / S
    cx = (cell_j + bx) * cell_w
    cy = (cell_i + by) * cell_h
    w = bw * img_size
    h = bh * img_size
    x1 = max(0, cx - w / 2)
    y1 = max(0, cy - h / 2)
    x2 = min(img_size, cx + w / 2)
    y2 = min(img_size, cy + h / 2)
    return [x1, y1, x2, y2], (cx, cy)

# ---------------------------------------------------------
# Load CSV and filter test set
# ---------------------------------------------------------
df = pd.read_csv(csv_path)
test_df = df[df["split_group"] == "test"].reset_index(drop=True)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = TinyYoloSingle(S=GRID_S, pretrained=False).to(device)
ckpt = torch.load(weights_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---------------------------------------------------------
# Transform
# ---------------------------------------------------------
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ---------------------------------------------------------
# Visualize hurricane trajectories
# ---------------------------------------------------------
for storm_id, grp in test_df.groupby(intl_col):
    centers_pred = []

    plt.figure(figsize=(8,6))
    plt.title(f"Storm {storm_id} Predicted Trajectory")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    for idx, row in grp.iterrows():
        img_path = os.path.join(image_folder, row["file_name"])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        input_t = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_t)[0]

        pc = pred[..., 0]
        flat_idx = torch.argmax(pc).item()
        cell_i = flat_idx // GRID_S
        cell_j = flat_idx % GRID_S
        pred_cell = pred[cell_i, cell_j].cpu().numpy()

        _, (cx, cy) = grid_to_bbox(pred_cell, cell_i, cell_j, GRID_S, IMG_SIZE)
        centers_pred.append((cx, cy))

        # draw predicted center on image
        ex = img.resize((IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(ex)
        r = 4
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill="red")
        ex.save(os.path.join(output_examples, f"{storm_id}_{idx}.png"))

    if centers_pred:
        centers_pred = np.array(centers_pred)
        plt.plot(centers_pred[:,0], centers_pred[:,1], "-o", color="red", label="Predicted center")
        plt.gca().invert_yaxis()  # y-axis points down in images
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_examples, f"trajectory_{storm_id}.png"))
        plt.close()
