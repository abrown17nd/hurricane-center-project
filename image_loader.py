import os
import re
import random
import pandas as pd
from PIL import Image, ImageDraw

# =========================================================
# CONFIGURATION
# =========================================================
LABELS_CSV = "data_from_tien/satellite_2015-2023/satellite/tc_labels.csv"
DOMAIN_FILE = "data_from_tien/satellite_2015-2023/satellite/domain.txt"
IMAGES_DIR = "data_from_tien/satellite_2015-2023/satellite/TC_biendong_625x500-2015-2023_Himawari"
BEST_TRACK_CSV = "data_from_tien/RSMC_Best_Track_Data_2024.csv"
DEBUG_MODE = True
IMG_WIDTH = 625
IMG_HEIGHT = 500

# =========================================================
# LOAD LABELS
# =========================================================
def load_labels():
    df = pd.read_csv(
        LABELS_CSV,
        header=None,
        names=["file_name", "x_min", "y_min", "x_max", "y_max", "intensity"]
    )
    pattern = re.compile(
        r"(?P<storm_id>\d+[A-Z]?)\.(?P<storm_name>[A-Z\-]+)\.(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\.(?P<hour>\d{4})"
    )
    parsed_data = df["file_name"].apply(
        lambda f: pattern.search(f).groupdict() if pattern.search(f) else {}
    )
    parsed_df = pd.DataFrame(parsed_data.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H%M",
        errors="coerce"
    )
    if DEBUG_MODE:
        df = df.sample(50, random_state=42).reset_index(drop=True)
    return df

# =========================================================
# LOAD DOMAIN
# =========================================================
def load_domain():
    domain = {}
    with open(DOMAIN_FILE, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=")
                domain[key.strip()] = float(val)
    return domain

def pixel_to_latlon(x, y, domain):
    start_lat = domain["biendong_startlat"]
    end_lat = domain["biendong_endlat"]
    start_lon = domain["biendong_startlon"]
    end_lon = domain["biendong_endlon"]
    lon = start_lon + (x / IMG_WIDTH) * (end_lon - start_lon)
    lat = end_lat - (y / IMG_HEIGHT) * (end_lat - start_lat)
    return lat, lon

def latlon_to_pixel(lat, lon, domain):
    start_lat = domain["biendong_startlat"]
    end_lat = domain["biendong_endlat"]
    start_lon = domain["biendong_startlon"]
    end_lon = domain["biendong_endlon"]
    x = (lon - start_lon) / (end_lon - start_lon) * IMG_WIDTH
    y = (end_lat - lat) / (end_lat - start_lat) * IMG_HEIGHT
    return x, y

# =========================================================
# LOAD BEST TRACK
# =========================================================
def load_best_track():
    best_track = pd.read_csv(BEST_TRACK_CSV)
    best_track.columns = best_track.columns.str.strip()
    short_map = {
        "International number ID": "ID",
        "Name of the storm": "name",
        "Time of analysis": "time",
        "Grade": "grade",
        "Latitude of the center": "lat",
        "Longitude of the center": "lon",
        "Central pressure": "p_cen",
        "Maximum sustained wind speed": "vmax",
        "Direction of the longest radius of 50kt winds or greater": "r50_dir",
        "The longest radius of 50kt winds or greater": "r50_long",
        "The shortest radius of 50kt winds or greater": "r50_short",
        "Direction of the longest radius of 30kt winds or greater": "r30_dir",
        "The longest radius of 30kt winds or greater": "r30_long",
        "The shortest radius of 30kt winds or greater": "r30_short",
        "Indicator of landfall or passage": "landfall"
    }
    best_track = best_track.rename(columns=short_map)
    if "time" in best_track.columns:
        best_track["datetime"] = pd.to_datetime(best_track["time"], errors="coerce")
    if "name" in best_track.columns:
        best_track["storm_name"] = best_track["name"].str.upper().fillna("")
    else:
        best_track["storm_name"] = ""
    return best_track

# =========================================================
# OVERLAY BBOX AND BEST TRACK
# =========================================================
def overlay_bbox_besttrack(image_name, df, domain, best_track):
    row = df[df["file_name"] == image_name]
    if row.empty:
        return None

    row = row.iloc[0]
    img_path = os.path.join(IMAGES_DIR, image_name)
    if not os.path.exists(img_path):
        return None

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw bounding box
    draw.rectangle(
        [row["x_min"], row["y_min"], row["x_max"], row["y_max"]],
        outline="red",
        width=2
    )

    # Match best track record by storm name and datetime
    subset = best_track[best_track["storm_name"].str.contains(str(row["storm_name"]), na=False)]
    if "datetime" in best_track.columns and pd.notna(row["datetime"]) and not subset.empty:
        subset = subset.copy()
        subset["time_diff"] = abs(subset["datetime"] - row["datetime"])
        subset = subset.sort_values("time_diff").head(1)
    if subset.empty:
        return img

    bt_row = subset.iloc[0]
    x_bt, y_bt = latlon_to_pixel(bt_row["lat"], bt_row["lon"], domain)
    r = 8  # radius of circle
    draw.ellipse([x_bt - r, y_bt - r, x_bt + r, y_bt + r], outline="yellow", width=2)

    return img
