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

sample_folder = "training_and_testing_samples"
csv_path = os.path.join(sample_folder, "sample_metadata.csv")
weights_path = "best_tiny_yolo_single.pth"

output_examples = os.path.join(sample_folder, "prediction_examples")
os.makedirs(output_examples, exist_ok=True)

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
def grid_to_bbox(pred_cell, cell_i, cell_j, S, img_size=448):
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

    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-9)

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(csv_path)

# ---------------------------------------------------------
# Load Model
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
# Inference Loop
# ---------------------------------------------------------
ious = []
example_counter = 0

for _, row in df.iterrows():
    img_path = os.path.join(sample_folder, row["file_name"])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    input_t = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_t)[0]

    # best cell
    pc = pred[..., 0]
    flat_idx = torch.argmax(pc).item()
    cell_i = flat_idx // GRID_S
    cell_j = flat_idx % GRID_S
    pred_cell = pred[cell_i, cell_j].cpu().numpy()

    pred_box = grid_to_bbox(pred_cell, cell_i, cell_j, GRID_S, IMG_SIZE)

    # ground truth box scaled to IMG_SIZE
    sx = IMG_SIZE / IMG_WIDTH
    sy = IMG_SIZE / IMG_HEIGHT
    gt_box = [
        row["x1"] * sx,
        row["y1"] * sy,
        row["x2"] * sx,
        row["y2"] * sy
    ]

    iou = compute_iou(pred_box, gt_box)
    ious.append(iou)

    # save example annotated images
    if example_counter < 30:
        ex = img.resize((IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(ex)
        draw.rectangle(pred_box, outline="red", width=3)
        draw.rectangle(gt_box, outline="green", width=3)
        ex.save(os.path.join(output_examples, f"sample_example_{example_counter}.png"))
        example_counter += 1

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
ious = np.array(ious)
print("Num samples:", len(ious))
print("Mean IoU:", np.mean(ious))
print("Median IoU:", np.median(ious))
print("IoU > 0.5:", np.mean(ious > 0.5))

# ---------------------------------------------------------
# Plot histogram
# ---------------------------------------------------------
plt.figure(figsize=(7,5))
plt.hist(ious, bins=20, edgecolor="black")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.title("IoU Distribution on 30 Random Samples")
plt.savefig(os.path.join(sample_folder, "iou_histogram_samples.png"), dpi=200)
plt.close()
