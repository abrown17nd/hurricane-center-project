import os
import csv
import tkinter as tk
from PIL import ImageTk
from image_loader import load_labels, load_domain, load_best_track, overlay_bbox_besttrack

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
# LOAD DATA
# =========================================================
df = load_labels()
domain = load_domain()
best_track = load_best_track()

# =========================================================
# LOAD EXISTING LABELS
# =========================================================
label_dict = {}
if os.path.exists(OUTPUT_LABEL_FILE):
    with open(OUTPUT_LABEL_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                label_dict[parts[0]] = parts[1]

# =========================================================
# IMAGE LIST
# =========================================================
image_list = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
image_list.sort()
if DEBUG_MODE:
    print(f"DEBUG: Loaded {len(image_list)} images from {IMAGES_DIR}")

# =========================================================
# SAVE LABELS
# =========================================================
def save_labels():
    with open(OUTPUT_LABEL_FILE, "w") as f:
        for k, v in label_dict.items():
            f.write(f"{k},{v}\n")
    with open(OUTPUT_LABELS_CSV, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for k, v in label_dict.items():
            writer.writerow([k, v])

# =========================================================
# GUI SETUP
# =========================================================
window = tk.Tk()
window.title("Image Labeling Tool")
current_index = 0
photo_cache = None  # persistent reference for ImageTk.PhotoImage

img_label = tk.Label(window)
img_label.pack()

info_label = tk.Label(window, text="", font=("Arial", 12))
info_label.pack(pady=5)

# =========================================================
# UPDATE IMAGE WITH OVERLAY
# =========================================================
def update_image():
    global current_index, photo_cache
    current_index = max(0, min(current_index, len(image_list)-1))

    image_name = image_list[current_index]
    if DEBUG_MODE:
        print(f"DEBUG: Showing image {current_index+1}/{len(image_list)} -> {image_name}")

    img = overlay_bbox_besttrack(image_name, df, domain, best_track)
    if img is None:
        print(f"DEBUG: overlay_bbox_besttrack returned None for {image_name}")
        return

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    photo_cache = ImageTk.PhotoImage(img)  # store reference
    img_label.config(image=photo_cache)

    status = label_dict.get(image_name, "unlabeled")
    info_label.config(text=f"{image_name}   |   Label: {status}")

# =========================================================
# MARK GOOD / BAD
# =========================================================
def mark_good(event=None):
    filename = image_list[current_index]
    print(f"DEBUG: Marking {filename} as good")
    label_dict[filename] = "good"
    save_labels()
    next_image()

def mark_bad(event=None):
    filename = image_list[current_index]
    print(f"DEBUG: Marking {filename} as bad")
    label_dict[filename] = "bad"
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
