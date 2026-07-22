"""
Autonomous driving scene captioner — NuPlan and nuScenes.

Iterates sensor frames, extracts front-cam image + structured annotations,
sends to a local MLLM (Qwen2-VL / InternVL2) or Claude API,
saves scene descriptions to a JSONL file.

NuPlan usage:
    python caption_scenes.py \
        --dataset     nuplan \
        --data_root   /proj/.../data \
        --split_dir   splits/mini \
        --blob_subdir sensor_blobs_mini \
        --camera      CAM_F0 \
        --model       /path/to/Qwen2-VL-7B-Instruct \
        --output      captions_nuplan.jsonl \
        --stride      10 --max_dbs 1

nuScenes usage:
    python caption_scenes.py \
        --dataset      nuscenes \
        --data_root    /proj/.../data_nusc \
        --nusc_version v1.0-mini \
        --camera       CAM_FRONT \
        --model        /path/to/Qwen2-VL-7B-Instruct \
        --output       captions_nusc.jsonl \
        --stride       1
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _degrees_to_compass(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


def build_annotation_text(ego: dict, objects: list[dict], tl_statuses: list[str]) -> str:
    lines = [
        f"EGO: {ego.get('speed_kph', 0):.1f} kph heading "
        f"{ego.get('heading_deg', 0):.1f}° ({ego.get('heading_compass', '?')})",
    ]
    if tl_statuses:
        lines.append(f"Traffic lights visible: {', '.join(tl_statuses)}")

    by_cat: dict[str, list] = {}
    for o in objects:
        by_cat.setdefault(o["category"], []).append(o)

    for cat, items in sorted(by_cat.items()):
        details = [
            f"{i['dist_m']}m {i['longitudinal']}-{i['lateral']} ({i['speed_ms']:.1f}m/s)"
            for i in items[:6]
        ]
        lines.append(f"  {cat} x{len(items)}: {', '.join(details)}")

    return "\n".join(lines)


def _world_to_ego(wx, wy, ego_x, ego_y, ego_yaw):
    cos_h, sin_h = math.cos(-ego_yaw), math.sin(-ego_yaw)
    dx, dy = wx - ego_x, wy - ego_y
    rel_x = cos_h * dx - sin_h * dy
    rel_y = sin_h * dx + cos_h * dy
    return rel_x, rel_y


# ---------------------------------------------------------------------------
# NuPlan helpers
# ---------------------------------------------------------------------------

def find_db_files(data_root: Path, split_dir: str) -> list[Path]:
    return sorted((data_root / split_dir).glob("*.db"))


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def nuplan_get_lidarpc_tokens(conn: sqlite3.Connection, stride: int = 1) -> list[bytes]:
    rows = conn.execute("SELECT token FROM lidar_pc ORDER BY timestamp ASC").fetchall()
    return [r["token"] for r in rows[::stride]]


def nuplan_get_ego_pose(conn: sqlite3.Connection, lidarpc_token: bytes) -> dict:
    row = conn.execute(
        """SELECT ep.x, ep.y, ep.vx, ep.vy, ep.qw, ep.qx, ep.qy, ep.qz
           FROM lidar_pc lp
           JOIN ego_pose ep ON ep.token = lp.ego_pose_token
           WHERE lp.token = ?""",
        (lidarpc_token,),
    ).fetchone()
    if row is None:
        return {}
    speed = math.hypot(row["vx"], row["vy"])
    yaw = _quat_to_yaw(row["qw"], row["qx"], row["qy"], row["qz"])
    heading_deg = math.degrees(yaw) % 360
    return {
        "x": row["x"], "y": row["y"], "yaw": yaw,
        "speed_ms": round(speed, 2),
        "speed_kph": round(speed * 3.6, 1),
        "heading_deg": round(heading_deg, 1),
        "heading_compass": _degrees_to_compass(heading_deg),
    }


def nuplan_get_objects(conn: sqlite3.Connection, lidarpc_token: bytes, ego: dict) -> list[dict]:
    rows = conn.execute(
        """SELECT lb.x, lb.y, lb.vx, lb.vy, lb.width, lb.length, cat.name AS category
           FROM lidar_box lb
           JOIN track t      ON t.token   = lb.track_token
           JOIN category cat ON cat.token = t.category_token
           WHERE lb.lidar_pc_token = ?""",
        (lidarpc_token,),
    ).fetchall()
    objs = []
    for r in rows:
        rel_x, rel_y = _world_to_ego(r["x"], r["y"], ego["x"], ego["y"], ego["yaw"])
        dist = math.hypot(rel_x, rel_y)
        objs.append({
            "category": r["category"],
            "dist_m": round(dist, 1),
            "rel_x": round(rel_x, 1),
            "rel_y": round(rel_y, 1),
            "lateral": "left" if rel_y > 0 else "right",
            "longitudinal": "ahead" if rel_x > 0 else "behind",
            "width": round(r["width"], 1),
            "length": round(r["length"], 1),
            "speed_ms": round(math.hypot(r["vx"], r["vy"]), 2),
        })
    objs.sort(key=lambda o: o["dist_m"])
    return objs


def nuplan_get_image(
    conn: sqlite3.Connection,
    data_root: Path,
    db_stem: str,
    lidarpc_token: bytes,
    camera: str,
    blob_subdir: str,
    debug: bool = False,
) -> Optional[Image.Image]:
    lp_ts = conn.execute(
        "SELECT timestamp FROM lidar_pc WHERE token = ?", (lidarpc_token,)
    ).fetchone()
    if lp_ts is None:
        return None
    row = conn.execute(
        """SELECT im.filename_jpg FROM image im
           JOIN camera cam ON cam.token = im.camera_token
           WHERE cam.channel = ?
           ORDER BY ABS(im.timestamp - ?) LIMIT 1""",
        (camera, lp_ts["timestamp"]),
    ).fetchone()
    if debug:
        print(f"[DEBUG] lidar ts={lp_ts['timestamp']} camera={camera} row={dict(row) if row else None}")
    if row is None:
        return None
    fname = row["filename_jpg"]
    candidates = [
        data_root / blob_subdir / fname,
        data_root / fname,
        data_root / blob_subdir / db_stem / camera / Path(fname).name,
    ]
    if debug:
        for c in candidates:
            print(f"[DEBUG] {c} exists={c.exists()}")
    for p in candidates:
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def nuplan_get_traffic_lights(conn: sqlite3.Connection, lidarpc_token: bytes) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT status FROM traffic_light_status WHERE lidar_pc_token = ?",
        (lidarpc_token,),
    ).fetchall()
    return [r["status"] for r in rows]


def process_nuplan_db(
    db_path: Path, data_root: Path, blob_subdir: str,
    generate_fn, camera: str, stride: int,
    output_file, already_done: set[str], debug: bool = False,
):
    conn = open_db(db_path)
    db_stem = db_path.stem
    tokens = nuplan_get_lidarpc_tokens(conn, stride)
    skipped = 0

    for i, token in enumerate(tqdm(tokens, desc=db_stem, leave=False)):
        token_hex = token.hex() if isinstance(token, bytes) else token
        key = f"nuplan__{db_stem}__{token_hex}"
        if key in already_done:
            continue

        ego = nuplan_get_ego_pose(conn, token)
        if not ego:
            continue

        image = nuplan_get_image(conn, data_root, db_stem, token, camera, blob_subdir,
                                 debug=(debug and i == 0))
        if image is None:
            skipped += 1
            continue

        objects = nuplan_get_objects(conn, token, ego)
        tl = nuplan_get_traffic_lights(conn, token)
        ann_text = build_annotation_text(ego, objects, tl)
        ego_out = {k: v for k, v in ego.items() if k != "yaw"}

        try:
            description = generate_fn(image, ann_text)
        except Exception as e:
            description = f"ERROR: {e}"

        output_file.write(json.dumps({
            "key": key, "dataset": "nuplan", "log": db_stem,
            "lidarpc_token": token_hex, "ego": ego_out,
            "objects": objects, "traffic_lights": tl,
            "annotation_text": ann_text, "description": description,
        }) + "\n")
        output_file.flush()

    if skipped:
        print(f"  [{db_stem}] skipped {skipped}/{len(tokens)} frames (image not found)")
    conn.close()


# ---------------------------------------------------------------------------
# nuScenes helpers
# ---------------------------------------------------------------------------

def _load_nusc_tables(data_root: Path, version: str) -> dict:
    """Load nuScenes JSON tables and build devkit-style derived fields."""
    base = data_root / version
    tables = {}
    for name in ["scene", "sample", "sample_data", "sample_annotation",
                  "ego_pose", "calibrated_sensor", "sensor", "category", "instance"]:
        path = base / f"{name}.json"
        rows = json.loads(path.read_text())
        tables[name] = {r["token"]: r for r in rows}
        tables[f"{name}_list"] = rows

    # Build sample["data"] = {channel: sd_token} and sample["anns"] = [ann_token, ...]
    # (these exist in the devkit but not in raw JSON files)
    for s in tables["sample"].values():
        s["data"] = {}
        s["anns"] = []

    for sd in tables["sample_data_list"]:
        if not sd.get("is_key_frame"):
            continue
        s = tables["sample"].get(sd["sample_token"])
        if s is None:
            continue
        cal = tables["calibrated_sensor"].get(sd["calibrated_sensor_token"], {})
        sensor = tables["sensor"].get(cal.get("sensor_token", ""), {})
        channel = sensor.get("channel", "")
        if channel:
            s["data"][channel] = sd["token"]

    for ann in tables["sample_annotation_list"]:
        s = tables["sample"].get(ann["sample_token"])
        if s is not None:
            s["anns"].append(ann["token"])

    return tables


def nusc_get_objects(tables: dict, sample: dict, ego_x, ego_y, ego_yaw) -> list[dict]:
    objs = []
    cat_by_token = {r["token"]: r["name"] for r in tables["category_list"]}
    inst_table = tables["instance"]

    for ann_token in sample["anns"]:
        ann = tables["sample_annotation"][ann_token]
        wx, wy = ann["translation"][0], ann["translation"][1]
        rel_x, rel_y = _world_to_ego(wx, wy, ego_x, ego_y, ego_yaw)
        dist = math.hypot(rel_x, rel_y)

        # velocity: from prev/next annotation positions if available
        speed = 0.0
        if ann.get("next") and ann["next"] in tables["sample_annotation"]:
            ann_next = tables["sample_annotation"][ann["next"]]
            # get timestamps from linked sample_data — approximate with 0.5s
            dx = ann_next["translation"][0] - ann["translation"][0]
            dy = ann_next["translation"][1] - ann["translation"][1]
            speed = round(math.hypot(dx, dy) / 0.5, 2)  # ~2Hz keyframes

        inst = inst_table.get(ann["instance_token"], {})
        cat_name = cat_by_token.get(inst.get("category_token", ""), ann.get("category_name", "unknown"))
        # nuScenes categories are hierarchical ("vehicle.car") — use top level
        cat_top = cat_name.split(".")[0]

        objs.append({
            "category": cat_top,
            "category_full": cat_name,
            "dist_m": round(dist, 1),
            "rel_x": round(rel_x, 1),
            "rel_y": round(rel_y, 1),
            "lateral": "left" if rel_y > 0 else "right",
            "longitudinal": "ahead" if rel_x > 0 else "behind",
            "width": round(ann["size"][0], 1),
            "length": round(ann["size"][1], 1),
            "speed_ms": speed,
        })
    objs.sort(key=lambda o: o["dist_m"])
    return objs


def process_nuscenes(
    data_root: Path, version: str,
    generate_fn, camera: str, stride: int,
    output_file, already_done: set[str],
    max_scenes: Optional[int] = None, debug: bool = False,
    shard: int = 0, num_shards: int = 1,
):
    print(f"Loading nuScenes tables ({version}) ...")
    tables = _load_nusc_tables(data_root, version)
    scenes = tables["scene_list"]
    if max_scenes:
        scenes = scenes[:max_scenes]
    scenes = scenes[shard::num_shards]
    print(f"Scenes this shard: {len(scenes)}")

    for scene in tqdm(scenes, desc="Scenes"):
        sample_token = scene["first_sample_token"]
        frame_idx = 0

        while sample_token:
            sample = tables["sample"][sample_token]

            if frame_idx % stride == 0 and camera in sample["data"]:
                cam_sd_token = sample["data"][camera]
                cam_sd = tables["sample_data"][cam_sd_token]
                ego_pose = tables["ego_pose"][cam_sd["ego_pose_token"]]

                key = f"nusc__{scene['name']}__{sample_token}"
                if key not in already_done:
                    img_path = data_root / cam_sd["filename"]
                    if img_path.exists():
                        image = Image.open(img_path).convert("RGB")

                        t = ego_pose["translation"]
                        r = ego_pose["rotation"]  # [w, x, y, z]
                        ego_yaw = _quat_to_yaw(r[0], r[1], r[2], r[3])
                        heading_deg = math.degrees(ego_yaw) % 360

                        # estimate ego speed from prev keyframe ego_pose if available
                        speed_ms = 0.0
                        if sample.get("prev") and sample["prev"] in tables["sample"]:
                            prev_sd_token = tables["sample"][sample["prev"]]["data"].get(camera)
                            if prev_sd_token and prev_sd_token in tables["sample_data"]:
                                prev_ep = tables["ego_pose"][
                                    tables["sample_data"][prev_sd_token]["ego_pose_token"]
                                ]
                                dt = (cam_sd["timestamp"] -
                                      tables["sample_data"][prev_sd_token]["timestamp"]) / 1e6
                                if dt > 0:
                                    dx = t[0] - prev_ep["translation"][0]
                                    dy = t[1] - prev_ep["translation"][1]
                                    speed_ms = math.hypot(dx, dy) / dt

                        ego = {
                            "x": t[0], "y": t[1],
                            "speed_ms": round(speed_ms, 2),
                            "speed_kph": round(speed_ms * 3.6, 1),
                            "heading_deg": round(heading_deg, 1),
                            "heading_compass": _degrees_to_compass(heading_deg),
                        }

                        objects = nusc_get_objects(tables, sample, t[0], t[1], ego_yaw)
                        ann_text = build_annotation_text(ego, objects, [])

                        try:
                            description = generate_fn(image, ann_text)
                        except Exception as e:
                            description = f"ERROR: {e}"

                        output_file.write(json.dumps({
                            "key": key, "dataset": "nuscenes",
                            "scene": scene["name"], "sample_token": sample_token,
                            "ego": {k: v for k, v in ego.items() if k not in ("x", "y")},
                            "objects": objects, "traffic_lights": [],
                            "annotation_text": ann_text, "description": description,
                        }) + "\n")
                        output_file.flush()

            sample_token = sample["next"]
            frame_idx += 1


# ---------------------------------------------------------------------------
# MLLM backends
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    if "Qwen2-VL" in model_name or "Qwen2VL" in model_name:
        return _load_qwen2vl(model_name)
    elif "InternVL" in model_name:
        return _load_internvl(model_name)
    elif model_name == "claude":
        return _load_claude_api()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _load_qwen2vl(model_name: str):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    def generate(image: Image.Image, annotation_text: str) -> str:
        prompt = _build_prompt(annotation_text)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return model, processor, generate


def _load_internvl(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def generate(image: Image.Image, annotation_text: str) -> str:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(model.device)
        response = model.chat(tokenizer, pixel_values, f"<image>\n{_build_prompt(annotation_text)}",
                              generation_config={"max_new_tokens": 512})
        return response.strip()

    return model, tokenizer, generate


def _load_claude_api():
    import anthropic, base64, io
    client = anthropic.Anthropic()

    def generate(image: Image.Image, annotation_text: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _build_prompt(annotation_text)},
            ]}],
        )
        return msg.content[0].text.strip()

    return None, None, generate


def _build_prompt(annotation_text: str) -> str:
    return (
        "You are analyzing a frame from an autonomous vehicle's front camera.\n\n"
        "Sensor annotations (ego-relative coordinates: x=forward, y=left):\n"
        f"{annotation_text}\n\n"
        "Write a concise but detailed scene description (4-6 sentences) covering:\n"
        "1. Environment type (urban street / highway / intersection / parking lot / etc.)\n"
        "2. Road layout visible: lanes, markings, traffic lights, signs\n"
        "3. All road users — their position, estimated distance, and behavior\n"
        "4. Ego state: speed, direction, current maneuver\n"
        "5. Scenario label: e.g. 'urban following', 'unprotected left turn', "
        "'highway lane change', 'pedestrian crossing', 'stationary in traffic'\n"
        "Be specific and factual. No speculation beyond what the image and annotations show."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      choices=["nuplan", "nuscenes"], default="nuplan")
    parser.add_argument("--data_root",    required=True, type=Path)
    parser.add_argument("--output",       required=True, type=Path)
    parser.add_argument("--model",        default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--stride",       type=int, default=10,
                        help="Sample every N frames (nuplan: lidarpc frames; nuscenes: keyframes)")
    # NuPlan-specific
    parser.add_argument("--split_dir",   default="splits/mini",
                        help="[nuplan] relative path to folder with .db files")
    parser.add_argument("--blob_subdir", default="sensor_blobs_mini",
                        help="[nuplan] image subfolder under data_root")
    parser.add_argument("--camera",      default=None,
                        help="Camera channel (nuplan: CAM_F0, nuscenes: CAM_FRONT). "
                             "Auto-set per dataset if omitted.")
    parser.add_argument("--max_dbs",     type=int, default=None,
                        help="[nuplan] max DB files to process (debug)")
    parser.add_argument("--shard",       type=int, default=0)
    parser.add_argument("--num_shards",  type=int, default=1)
    # nuScenes-specific
    parser.add_argument("--nusc_version", default="v1.0-mini",
                        help="[nuscenes] annotation version subfolder (v1.0-mini / v1.0-trainval)")
    parser.add_argument("--max_scenes",  type=int, default=None,
                        help="[nuscenes] max scenes to process (debug)")
    args = parser.parse_args()

    # Default camera per dataset
    if args.camera is None:
        args.camera = "CAM_F0" if args.dataset == "nuplan" else "CAM_FRONT"

    # Resume
    already_done: set[str] = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["key"])
                except Exception:
                    pass
        print(f"Resuming — {len(already_done)} frames already done.")

    print(f"Loading model: {args.model} ...")
    _, _, generate_fn = load_model(args.model)
    print("Model ready.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "nuplan":
        db_files = find_db_files(args.data_root, args.split_dir)
        if not db_files:
            raise FileNotFoundError(f"No .db files in {args.data_root / args.split_dir}")
        print(f"Found {len(db_files)} DB files")
        if args.max_dbs:
            db_files = db_files[: args.max_dbs]
        db_files = db_files[args.shard :: args.num_shards]
        print(f"This shard: {len(db_files)} DB files")

        with open(args.output, "a") as out:
            for db_path in tqdm(db_files, desc="Logs"):
                process_nuplan_db(
                    db_path, args.data_root, args.blob_subdir,
                    generate_fn, args.camera, args.stride, out, already_done,
                    debug=args.max_dbs is not None,
                )

    else:  # nuscenes
        with open(args.output, "a") as out:
            process_nuscenes(
                args.data_root, args.nusc_version,
                generate_fn, args.camera, args.stride, out, already_done,
                max_scenes=args.max_scenes, debug=args.max_scenes is not None,
                shard=args.shard, num_shards=args.num_shards,
            )

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
