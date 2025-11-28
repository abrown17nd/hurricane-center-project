import os
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

# =========================================================
# PIXEL <-> LAT/LON
# =========================================================
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
