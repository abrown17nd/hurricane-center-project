#!/usr/bin/env python3
# visualize_dino_val_test_selected.py
"""
Visualize DINO MLP predictions for validation and test images.
Ground truth boxes in green, predicted centers in red.
Features:
- Saves 5 randomly selected starting images per split to 'selected_images' folder.
- Uses 'selected_images' folder if it exists.
- Outputs:
    pred_examples/val/
    pred_examples/test/
"""

import os
import argparse
import random
import shutil
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import pandas as pd
from torch import nn

# ---------------- MODEL DEFINITION ----------------
class DINO_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[2048, 512], output_dim=2):
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

# ---------------- DATASET ----------------
class SingleObjectMLPDataset(torch.utils.data.Dataset):
    def __init__(self, df, folder_path, img_width=625, img_height=500, downsample=2, transform=None,
                 col_file="file_name", col_x1="x1", col_x2="x2", col_y1="y1", col_y2="y2"):
        self.df = df.reset_index(drop=True)
        self.folder_path = folder_path
        self.transform = transform
        self.img_w = img_width // downsample
        self.img_h = img_height // downsample
        self.downsample = downsample
        self.COL_FILE = col_file
        self.COL_X1 = col_x1
        self.COL_X2 = col_x2
        self.COL_Y1 = col_y1
        self.COL_Y2 = col_y2

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.folder_path, str(row[self.COL_FILE]).strip())
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_w, self.img_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        img_flat = img.view(-1)
        cx, cy = float(row[self.COL_X1]+row[self.COL_X2])/2.0, float(row[self.COL_Y1]+row[self.COL_Y2])/2.0
        target = torch.tensor([cx, cy], dtype=torch.float32)
        return img_flat, target

# ---------------- VISUALIZATION ----------------
def visualize_examples(model, dataset, device, out_folder, num_examples=5):
    os.makedirs(out_folder, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    with torch.no_grad():
        for idx in indices:
            img_flat, target = dataset[idx]
            pred = model(img_flat.unsqueeze(0).to(device))[0].cpu().numpy()

            row = dataset.df.iloc[idx]
            fname = str(row["file_name"]).strip()
            img_path = os.path.join(dataset.folder_path, fname)
            img = Image.open(img_path).convert("RGB")

            # Draw ground truth box
            gt_box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            draw = ImageDraw.Draw(img)
            draw.rectangle(gt_box, outline="green", width=2)

            # Draw predicted center
            pred_x, pred_y = pred
            r = 5
            draw.ellipse([pred_x - r, pred_y - r, pred_x + r, pred_y + r], fill="red")

            out_path = os.path.join(out_folder, fname)
            img.save(out_path)

# ---------------- SELECT STARTING IMAGES ----------------
def save_selected_images(df, images_folder, selected_folder, split, num_images=5):
    os.makedirs(selected_folder, exist_ok=True)
    split_df = df[df["split_group"]==split].reset_index(drop=True)
    if split_df.empty:
        return
    selected_indices = random.sample(range(len(split_df)), min(num_images, len(split_df)))
    for idx in selected_indices:
        fname = str(split_df.iloc[idx]["file_name"]).strip()
        src_path = os.path.join(images_folder, fname)
        dst_path = os.path.join(selected_folder, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

# ---------------- MAIN ----------------
def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.csv)

    # Determine folder to use
    images_folder = args.images
    selected_folder = "selected_images"
    if os.path.exists(selected_folder) and os.listdir(selected_folder):
        print(f"Using existing images from '{selected_folder}'")
        images_folder = selected_folder
    else:
        print(f"Saving starting images to '{selected_folder}'")
        for split in ["val", "test"]:
            save_selected_images(df, args.images, selected_folder, split, num_images=5)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                                         std=[0.229,0.224,0.225])])

    model = None

    for split in ["val", "test"]:
        split_df = df[df["split_group"]==split].reset_index(drop=True)
        if split_df.empty:
            print(f"No images found for split '{split}'")
            continue

        dataset = SingleObjectMLPDataset(split_df, images_folder, transform=transform)

        if model is None:
            input_dim = dataset[0][0].numel()
            model = DINO_MLP(input_dim=input_dim).to(DEVICE)
            model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))

        out_folder = os.path.join(args.out_folder, split)
        visualize_examples(model, dataset, DEVICE, out_folder, num_examples=5)
        print(f"{split.capitalize()} predictions saved to {os.path.abspath(out_folder)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Visualize DINO MLP predictions for val/test splits with selected images")
    parser.add_argument("--csv", type=str, default="random_groups.csv", help="CSV file with split_group column")
    parser.add_argument("--images", type=str, default="images/TC_biendong_625x500-2015-2023_Himawari", help="Folder containing input images")
    parser.add_argument("--ckpt", type=str, default="dino_mlp_model.pth", help="Trained model checkpoint")
    parser.add_argument("--out_folder", type=str, default="pred_examples", help="Output folder for prediction images")
    args = parser.parse_args()
    main(args)
