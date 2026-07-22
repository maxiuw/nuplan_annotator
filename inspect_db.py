"""Quick inspector — run on Berzelius to understand DB structure before captioning."""
import sqlite3, sys, pathlib

data_root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")

# Find first db file anywhere under data_root
dbs = list(data_root.rglob("*.db"))
if not dbs:
    print("No .db files found under", data_root)
    sys.exit(1)

db = dbs[0]
print(f"Using: {db}\n")
conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row

# Tables
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

# Sample image filenames
print("\n--- image.filename_jpg samples ---")
for r in conn.execute("SELECT filename_jpg FROM image LIMIT 5").fetchall():
    print(" ", r[0])

# Ego pose sample
print("\n--- ego_pose sample ---")
r = conn.execute("SELECT x,y,vx,vy,heading FROM ego_pose LIMIT 1").fetchone()
if r:
    print(dict(r))

# Lidar box / tracked objects
print("\n--- lidar_box + category sample ---")
rows = conn.execute("""
    SELECT lb.x, lb.y, lb.vx, lb.vy, c.name
    FROM lidar_box lb JOIN category c ON c.token=lb.category_token LIMIT 5
""").fetchall()
for r in rows:
    print(dict(r))

# Camera channels available
print("\n--- camera channels ---")
for r in conn.execute("SELECT DISTINCT channel FROM camera").fetchall():
    print(" ", r[0])

# Count frames
n = conn.execute("SELECT COUNT(*) FROM lidar_pc").fetchone()[0]
print(f"\nTotal lidar_pc frames: {n}")
conn.close()
