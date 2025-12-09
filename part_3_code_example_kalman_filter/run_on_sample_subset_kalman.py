import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn as nn
from torchvision import models
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
GRID_S = 7
IMG_SIZE = 448
IMG_WIDTH = 625
IMG_HEIGHT = 500

subset_folder = "training_and_testing_samples_kalman/"
subset_csv = os.path.join(subset_folder, "subset_labels.csv")

weights_path = "best_tiny_yolo_single.pth"
output_examples = "subset_predictions/"
os.makedirs(output_examples, exist_ok=True)

storm_col = "International number ID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Utility
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
# Load subset CSV
# ---------------------------------------------------------
df = pd.read_csv(subset_csv)

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
# Loop for trajectories and predicted frames
# ---------------------------------------------------------
for storm_id, grp in df.groupby(storm_col):
    centers = []

    plt.figure(figsize=(7, 6))
    plt.title("Storm {} - Predicted trajectory".format(storm_id))
    plt.xlabel("X")
    plt.ylabel("Y")

    storm_output_images = []

    for idx, row in grp.iterrows():
        img_path = os.path.join(subset_folder, row["file_name"])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        inp = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inp)[0]

        pc = pred[..., 0]
        flat = torch.argmax(pc).item()
        ci = flat // GRID_S
        cj = flat % GRID_S
        pred_cell = pred[ci, cj].cpu().numpy()

        _, (cx, cy) = grid_to_bbox(pred_cell, ci, cj, GRID_S)
        centers.append((cx, cy))

        ex = img.resize((IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(ex)
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")

        out_name = "{}_{}.png".format(storm_id, idx)
        ex.save(os.path.join(output_examples, out_name))
        storm_output_images.append(out_name)

    # trajectory plot
    if centers:
        centers = np.array(centers)
        plt.plot(centers[:, 0], centers[:, 1], "-o", color="red")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_examples, "traj_{}.png".format(storm_id)))
        plt.close()

    # ---------------------------------------------------------
    # GIF creation
    # ---------------------------------------------------------
    gif_folder = "subset_gifs/"
    os.makedirs(gif_folder, exist_ok=True)

    frames = [Image.open(os.path.join(output_examples, f)) for f in sorted(storm_output_images)]

    if frames:
        gif_path = os.path.join(gif_folder, "storm_{}.gif".format(storm_id))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.axis("off")
        img_show = ax.imshow(frames[0])

        def update(i):
            ax.clear()
            ax.axis("off")
            ax.imshow(frames[i])
            return [img_show]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=500)
        anim.save(gif_path, writer=PillowWriter(fps=2))
        plt.close(fig)

        print("Saved GIF:", gif_path)
