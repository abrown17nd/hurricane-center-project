import os
import math
import pandas as pd
from tkinter import Tk, Label, Canvas
from PIL import Image, ImageTk, ImageDraw

# =========================================================
# PARAMETERS
# =========================================================
csv_source = "joined_filtered_labels_images_besttrack.csv"
folder_path = r"data_from_tien\satellite_2015-2023\satellite\TC_biendong_625x500-2015-2023_Himawari"
output_marks_file = "image_quality_marks.csv"

img_width = 625
img_height = 500

biendong_startlat = -4.99
biendong_startlon = 90.01
biendong_endlat   = 34.99
biendong_endlon   = 134.49

# =========================================================
# LOAD CSV  +  DEBUG
# =========================================================
print("DEBUG: Loading CSV:", csv_source)
df = pd.read_csv(csv_source)

print("DEBUG: CSV loaded. Number of rows:", len(df))
print("DEBUG: Columns:", list(df.columns))

missing_names = df["file_name"].isna().sum()
if missing_names > 0:
    print("DEBUG WARNING: Missing file_name entries:", missing_names)

# =========================================================
# PRECOMPUTED CONSTANTS
# =========================================================
R = 6378137.0

def lonlat_to_mercator_m(lon_deg, lat_deg):
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    lat_rad = max(min(lat_rad, math.radians(85.05112878)), math.radians(-85.05112878))
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y

x_min_m, y_min_m = lonlat_to_mercator_m(biendong_startlon, biendong_startlat)
x_max_m, y_max_m = lonlat_to_mercator_m(biendong_endlon, biendong_endlat)

mean_lat = (biendong_startlat + biendong_endlat) / 2
cos_lat = math.cos(math.radians(mean_lat))

# =========================================================
# THREE PROJECTION FUNCTIONS
# =========================================================
def pix_linear(lat, lon):
    x = (lon - biendong_startlon) / (biendong_endlon - biendong_startlon) * img_width
    y = img_height - (lat - biendong_startlat) / (biendong_endlat - biendong_startlat) * img_height
    return int(x), int(y)

def pix_mercator(lat, lon):
    x_m, y_m = lonlat_to_mercator_m(lon, lat)
    nx = (x_m - x_min_m) / (x_max_m - x_min_m)
    ny = (y_m - y_min_m) / (y_max_m - y_min_m)
    return int(nx * img_width), int((1 - ny) * img_height)

def pix_coslat(lat, lon):
    nx = (lon - biendong_startlon) * cos_lat / ((biendong_endlon - biendong_startlon) * cos_lat)
    ny = (lat - biendong_startlat) / (biendong_endlat - biendong_startlat)
    return int(nx * img_width), int((1 - ny) * img_height)

# =========================================================
# MARK STORAGE
# =========================================================
marks = {}
if os.path.exists(output_marks_file):
    print("DEBUG: Loading existing marks file.")
    old = pd.read_csv(output_marks_file)
    for idx, r in old.iterrows():
        marks[r["image_name"]] = r["mark"]

# =========================================================
# GUI SETUP
# =========================================================
root = Tk()
root.title("TC Image Label Review – All Projections")

canvas = Canvas(root, width=img_width, height=img_height)
canvas.pack()

label_info = Label(root, text="", font=("Arial", 14))
label_info.pack()

index = 0
tk_img = None

# =========================================================
# DISPLAY FUNCTION  +  LONG-PATH FIX + DEBUG
# =========================================================
def show_image():
    global tk_img, index

    index_clamped = max(0, min(index, len(df) - 1))
    if index_clamped != index:
        print(f"DEBUG: index corrected from {index} to {index_clamped}")
        index = index_clamped

    row = df.iloc[index]

    print("\n==============================================")
    print(f"DEBUG row index: {index}")
    print(f"DEBUG raw file_name: {row['file_name']!r}")

    # Clean filename
    file_name_clean = str(row["file_name"]).strip()

    # -----------------------------------------------------
    # WINDOWS LONG PATH FIX
    # -----------------------------------------------------
    folder_abs = os.path.abspath(folder_path)
    if not folder_abs.startswith("\\\\?\\"):
        folder_lp = "\\\\?\\" + folder_abs
    else:
        folder_lp = folder_abs

    img_path = os.path.join(folder_lp, file_name_clean)

    print("DEBUG absolute folder:", folder_abs)
    print("DEBUG long-path folder:", folder_lp)
    print("DEBUG combined path:", img_path)
    print("DEBUG os.path.exists:", os.path.exists(img_path))

    if not os.path.exists(img_path):
        print("MISSING FILE:", img_path)
        return

    # -----------------------------------------------------
    # LOAD AND DRAW
    # -----------------------------------------------------
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.rectangle([(row.x1, row.y1), (row.x2, row.y2)], outline="red", width=3)

    lat = row["Latitude of the center"]
    lon = row["Longitude of the center"]
    r = 6

    px1, py1 = pix_linear(lat, lon)
    draw.ellipse((px1-r, py1-r, px1+r, py1+r), outline="red", width=18)

    px2, py2 = pix_mercator(lat, lon)
    draw.ellipse((px2-r, py2-r, px2+r, py2+r), outline="cyan", width=3)

    px3, py3 = pix_coslat(lat, lon)
    draw.ellipse((px3-r, py3-r, px3+r, py3+r), outline="magenta", width=3)

    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    label_info.config(
        text=f"{row['image_name']}   {index+1}/{len(df)}   Mark: {marks.get(row['image_name'], 'None')}"
    )

# =========================================================
# KEYBINDS
# =========================================================
def next_img(event=None):
    global index
    index += 1
    show_image()

def prev_img(event=None):
    global index
    index -= 1
    show_image()

def mark_good(event=None):
    row = df.iloc[index]
    marks[row.image_name] = "good"
    next_img()
    save_marks()
    show_image()

def mark_bad(event=None):
    row = df.iloc[index]
    marks[row.image_name] = "bad"
    next_img()
    save_marks()
    show_image()

def save_marks():
    out = pd.DataFrame([{"image_name": k, "mark": v} for k, v in marks.items()])
    out.to_csv(output_marks_file, index=False)

root.bind("<Right>", next_img)
root.bind("<Left>", prev_img)
root.bind("g", mark_good)
root.bind("b", mark_bad)

# =========================================================
# START GUI
# =========================================================
show_image()
root.mainloop()
