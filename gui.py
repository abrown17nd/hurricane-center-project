import os
import csv
import tkinter as tk
from PIL import Image, ImageTk
from image_loader import load_labels

# =========================================================
# CONFIGURATION
# =========================================================
IMAGES_DIR = "data_from_tien/satellite_2015-2023/satellite/TC_biendong_625x500-2015-2023_Himawari"
IMG_WIDTH = 625
IMG_HEIGHT = 500
DEBUG_MODE = True
OUTPUT_LABEL_FILE = "labels_output.txt"
OUTPUT_LABELS_CSV = "labels_output.csv"

# =========================================================
# LOAD LABELS
# =========================================================
df = load_labels()

label_dict = {}
if os.path.exists(OUTPUT_LABEL_FILE):
    with open(OUTPUT_LABEL_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]

# =========================================================
# BUILD IMAGE LIST
# =========================================================
image_list = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
image_list.sort()

if DEBUG_MODE:
    print(f"Loaded {len(image_list)} images from {IMAGES_DIR}")

# =========================================================
# SAVE LABELS
# =========================================================
def save_labels():
    with open(OUTPUT_LABEL_FILE, "w") as f:
        for k, v in label_dict.items():
            f.write(f"{k},{v}\n")
    with open(OUTPUT_LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for k, v in label_dict.items():
            writer.writerow([k, v])

# =========================================================
# GUI WINDOW SETUP
# =========================================================
window = tk.Tk()
window.title("Image Labeling Tool")
current_index = 0

img_label = tk.Label(window)
img_label.pack()

info_label = tk.Label(window, text="", font=("Arial", 12))
info_label.pack(pady=5)

# =========================================================
# UPDATE IMAGE
# =========================================================
def update_image():
    global current_index
    current_index = max(0, min(current_index, len(image_list)-1))

    image_path = os.path.join(IMAGES_DIR, image_list[current_index])
    img = Image.open(image_path).resize((IMG_WIDTH, IMG_HEIGHT))
    photo = ImageTk.PhotoImage(img)

    img_label.config(image=photo)
    img_label.image = photo

    filename = image_list[current_index]
    status = label_dict.get(filename, "unlabeled")
    info_label.config(text=f"{filename}   |   Label: {status}")

# =========================================================
# MARK GOOD / BAD
# =========================================================
def mark_good(event=None):
    label_dict[image_list[current_index]] = "good"
    save_labels()
    next_image()

def mark_bad(event=None):
    label_dict[image_list[current_index]] = "bad"
    save_labels()
    next_image()

# =========================================================
# NAVIGATION
# =========================================================
def next_image(event=None):
    global current_index
    current_index += 1
    if current_index >= len(image_list):
        current_index = len(image_list) - 1
    update_image()

def previous_image(event=None):
    global current_index
    current_index -= 1
    if current_index < 0:
        current_index = 0
    update_image()

# =========================================================
# UI BUTTONS
# =========================================================
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Good (G)", command=mark_good, bg="green", fg="white").grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Bad (B)", command=mark_bad, bg="red", fg="white").grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Previous", command=previous_image).grid(row=0, column=2, padx=5)
tk.Button(button_frame, text="Next", command=next_image).grid(row=0, column=3, padx=5)

# =========================================================
# KEYBOARD BINDS
# =========================================================
window.bind("g", mark_good)
window.bind("b", mark_bad)
window.bind("<Left>", previous_image)
window.bind("<Right>", next_image)

# =========================================================
# START
# =========================================================
update_image()
window.mainloop()
