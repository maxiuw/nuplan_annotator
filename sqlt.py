import sqlite3
db = "/proj/berzelius-2023-364/users/x_macwo/code/nuplan_annot/nuplan_annotator/data/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
conn = sqlite3.connect(db)

print("=== image table schema ===")
print(conn.execute("SELECT sql FROM sqlite_master WHERE name='image'").fetchone()[0])

print("\n=== filename_jpg samples ===")
for r in conn.execute("SELECT filename_jpg, hex(token) FROM image LIMIT 3"):
    print(r)

print("\n=== ego_pose_token join count ===")
q = "SELECT COUNT(*) FROM lidar_pc lp JOIN image im ON im.ego_pose_token = lp.ego_pose_token"
print(conn.execute(q).fetchone())

print("\n=== lidar_pc schema ===")
print(conn.execute("SELECT sql FROM sqlite_master WHERE name='lidar_pc'").fetchone()[0])
