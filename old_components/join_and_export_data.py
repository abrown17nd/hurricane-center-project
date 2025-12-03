import sqlite3
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================
db_path = r"tc_data.db"
export_csv = r"joined_filtered_labels_images_besttrack.csv"

# ==========================================================
# SQL QUERY
# ==========================================================

query = r"""
SELECT 
    t.image_name,
    t.x1, t.y1, t.x2, t.y2, t.category,
    h.file_name,
    h.storm_name,
    h.year, h.month, h.day, h.hour,
    b.[International number ID],
    b.[Name of the storm],
    b.[Time of analysis],
    b.Grade,
    b.[Latitude of the center],
    b.[Longitude of the center],
    b.[Central pressure],
    b.[Maximum sustained wind speed],
    b.[Direction of the longest radius of 50kt winds or greater],
    b.[The longest radius of 50kt winds or greater],
    b.[The shortest radius of 50kt winds or greater],
    b.[Direction of the longest radius of 30kt winds or greater],
    b.[The longest radius of 30kt winds or greater],
    b.[The shortest radius of 30kt winds or greater],
    b.[Indicator of landfall or passage]
FROM tc_labels t
INNER JOIN himawari_images h
    ON t.image_name LIKE h.file_name || '%'
INNER JOIN best_track b
    ON h.storm_name = b.[Name of the storm]
   AND datetime(
         h.year || '-' || 
         h.month || '-' || 
         h.day || ' ' ||
         substr(h.hour, 1, 2) || ':' ||
         substr(h.hour, 3, 2) || ':00'
       ) = b.[Time of analysis]
WHERE b.[Latitude of the center] BETWEEN -4.99 AND 34.99
  AND b.[Longitude of the center] BETWEEN 90.01 AND 134.49;
"""

# ==========================================================
# RUN QUERY AND EXPORT
# ==========================================================

conn = sqlite3.connect(db_path)
df = pd.read_sql_query(query, conn)
conn.close()

df.to_csv(export_csv, index=False)

print(f"Export complete. Rows written: {len(df)}")
print(f"CSV saved to: {export_csv}")
