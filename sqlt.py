import sqlite3
db = "/proj/berzelius-2023-364/users/x_macwo/code/nuplan_annot/nuplan_annotator/data/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

print("=== lidar_box schema ===")
row = conn.execute("SELECT sql FROM sqlite_master WHERE name='lidar_box'").fetchone()
print(row[0])

print("\n=== category table exists? ===")
row = conn.execute("SELECT sql FROM sqlite_master WHERE name='category'").fetchone()
print(row[0] if row else "NO category table")

print("\n=== lidar_box sample row ===")
r = conn.execute("SELECT * FROM lidar_box LIMIT 1").fetchone()
print(dict(r))
