import os
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
# LOAD FROM PREGENERATED CSV
# =========================================================
df = pd.read_csv(csv_source)

# =========================================================
# DOMAIN CONVERSION
# =========================================================
def latlon_to_pixels(lat, lon):
    x = (lon - biendong_startlon) / (biendong_endlon - biendong_startlon) * img_width
    y = img_height - (lat - biendong_startlat) / (biendong_endlat - biendong_startlat) * img_height
    return int(x), int(y)

# =========================================================
# MARK STORAGE
# =========================================================
marks = {}
if os.path.exists(output_marks_file):
    old = pd.read_csv(output_marks_file)
    for idx, r in old.iterrows():
        marks[r["image_name"]] = r["mark"]

# =========================================================
# GUI SETUP
# =========================================================
root = Tk()
root.title("TC Image Label Review")

canvas = Canvas(root, width=img_width, height=img_height)
canvas.pack()

label_info = Label(root, text="", font=("Arial", 14))
label_info.pack()

index = 0
tk_img = None

# =========================================================
# DISPLAY FUNCTION
# =========================================================
def show_image():
    global tk_img, index

    index = max(0, min(index, len(df) - 1))
    row = df.iloc[index]

    img_path = os.path.join(folder_path, row["file_name"])
    if not os.path.exists(img_path):
        print("Missing:", img_path)
        return

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Bounding box
    draw.rectangle([(row.x1, row.y1), (row.x2, row.y2)],
                   outline="red", width=3)

    # Best-track point
    px, py = latlon_to_pixels(row["Latitude of the center"],
                              row["Longitude of the center"])
    r = 6
    draw.ellipse((px - r, py - r, px + r, py + r),
                 outline="yellow", width=3)

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
    save_marks()
    show_image()

def mark_bad(event=None):
    row = df.iloc[index]
    marks[row.image_name] = "bad"
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
