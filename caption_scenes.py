"""
NuPlan scene captioner.

Iterates NuPlan DB files, extracts front-cam image + structured annotations,
sends to a local MLLM (Qwen2-VL or InternVL2), saves descriptions to jsonl.

Usage (mini, local test):
    python caption_scenes.py \
        --data_root /proj/.../data \
        --split_dir splits/mini \
        --output captions_mini.jsonl \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --stride 10 \
        --max_dbs 1       # remove for full run

Or with Claude API (no GPU needed):
    ANTHROPIC_API_KEY=sk-... python caption_scenes.py ... --model claude
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
# NuPlan DB helpers
# ---------------------------------------------------------------------------

def find_db_files(data_root: Path, split_dir: str) -> list[Path]:
    return sorted((data_root / split_dir).glob("*.db"))


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_lidarpc_tokens(conn: sqlite3.Connection, stride: int = 1) -> list[str]:
    rows = conn.execute("SELECT token FROM lidar_pc ORDER BY timestamp ASC").fetchall()
    return [r["token"] for r in rows[::stride]]


def get_ego_pose(conn: sqlite3.Connection, lidarpc_token: str) -> dict:
    row = conn.execute(
        """SELECT ep.vx, ep.vy, ep.qw, ep.qx, ep.qy, ep.qz
           FROM lidar_pc lp
           JOIN ego_pose ep ON ep.token = lp.ego_pose_token
           WHERE lp.token = ?""",
        (lidarpc_token,),
    ).fetchone()
    if row is None:
        return {}
    speed = math.hypot(row["vx"], row["vy"])
    # yaw from quaternion: atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    qw, qx, qy, qz = row["qw"], row["qx"], row["qy"], row["qz"]
    heading_rad = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    heading_deg = math.degrees(heading_rad) % 360
    return {
        "speed_ms": round(speed, 2),
        "speed_kph": round(speed * 3.6, 1),
        "heading_deg": round(heading_deg, 1),
        "heading_compass": _degrees_to_compass(heading_deg),
    }


def get_tracked_objects(conn: sqlite3.Connection, lidarpc_token: str) -> list[dict]:
    rows = conn.execute(
        """SELECT lb.x, lb.y, lb.vx, lb.vy, lb.width, lb.length, cat.name AS category
           FROM lidar_box lb
           JOIN category cat ON cat.token = lb.category_token
           WHERE lb.lidar_pc_token = ?""",
        (lidarpc_token,),
    ).fetchall()
    objs = []
    for r in rows:
        dist = math.hypot(r["x"], r["y"])
        # x=forward, y=left in ego frame
        lat = "left" if r["y"] > 0 else "right"
        lon = "ahead" if r["x"] > 0 else "behind"
        objs.append({
            "category": r["category"],
            "dist_m": round(dist, 1),
            "rel_x": round(r["x"], 1),   # positive = ahead
            "rel_y": round(r["y"], 1),   # positive = left
            "lateral": lat,
            "longitudinal": lon,
            "width": round(r["width"], 1),
            "length": round(r["length"], 1),
            "speed_ms": round(math.hypot(r["vx"], r["vy"]), 2),
        })
    objs.sort(key=lambda o: o["dist_m"])
    return objs


def get_camera_image(
    conn: sqlite3.Connection,
    data_root: Path,
    db_stem: str,
    lidarpc_token: str,
    camera: str = "CAM_F0",
    blob_subdir: str = "sensor_blobs_mini",
) -> Optional[Image.Image]:
    row = conn.execute(
        """SELECT im.filename_jpg, im.token
           FROM image im
           JOIN camera cam ON cam.token = im.camera_token
           WHERE im.lidar_pc_token = ? AND cam.channel = ?""",
        (lidarpc_token, camera),
    ).fetchone()
    if row is None:
        return None

    # Try candidate paths in order of likelihood
    candidates = []

    if row["filename_jpg"]:
        fname = row["filename_jpg"]
        # Might be full relative path from data_root
        candidates.append(data_root / fname)
        # Might be just the filename — construct from known structure
        candidates.append(data_root / blob_subdir / db_stem / camera / fname)
        # Same but filename might already include extension
        candidates.append(data_root / blob_subdir / db_stem / camera / Path(fname).name)

    # Fallback: use image token as filename (NuPlan sometimes stores token as hex)
    if row["token"]:
        tok = row["token"]
        if isinstance(tok, bytes):
            tok = tok.hex()
        candidates.append(data_root / blob_subdir / db_stem / camera / f"{tok}.jpg")

    for p in candidates:
        if p.exists():
            return Image.open(p).convert("RGB")

    return None


def get_traffic_lights(conn: sqlite3.Connection, lidarpc_token: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT status FROM traffic_light_status WHERE lidar_pc_token = ?",
        (lidarpc_token,),
    ).fetchall()
    return [r["status"] for r in rows]


def _degrees_to_compass(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


def build_annotation_text(ego: dict, objects: list[dict], tl_statuses: list[str]) -> str:
    lines = [
        f"EGO: {ego.get('speed_kph', 0):.1f} kph heading {ego.get('heading_deg', 0):.1f}° ({ego.get('heading_compass', '?')})",
    ]
    if tl_statuses:
        lines.append(f"Traffic lights visible: {', '.join(tl_statuses)}")

    by_cat: dict[str, list] = {}
    for o in objects:
        by_cat.setdefault(o["category"], []).append(o)

    for cat, items in sorted(by_cat.items()):
        details = []
        for i in items[:6]:
            details.append(
                f"{i['dist_m']}m {i['longitudinal']}-{i['lateral']} "
                f"({i['speed_ms']:.1f}m/s)"
            )
        lines.append(f"  {cat} x{len(items)}: {', '.join(details)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MLLM backends
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Returns (model, processor, generate_fn). Only generate_fn is used externally."""
    if "Qwen2-VL" in model_name or "Qwen2VL" in model_name:
        return _load_qwen2vl(model_name)
    elif "InternVL" in model_name:
        return _load_internvl(model_name)
    elif model_name == "claude":
        return _load_claude_api()
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: Qwen2-VL-*, InternVL*, claude")


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
        decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

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
        prompt = f"<image>\n{_build_prompt(annotation_text)}"
        response = model.chat(tokenizer, pixel_values, prompt, generation_config={"max_new_tokens": 512})
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
            model="claude-sonnet-4-6",
            max_tokens=512,
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
# Main pipeline
# ---------------------------------------------------------------------------

def process_db(
    db_path: Path,
    data_root: Path,
    blob_subdir: str,
    generate_fn,
    camera: str,
    stride: int,
    output_file,
    already_done: set[str],
):
    conn = open_db(db_path)
    db_stem = db_path.stem
    tokens = get_lidarpc_tokens(conn, stride=stride)
    skipped_no_image = 0

    for token in tqdm(tokens, desc=db_stem, leave=False):
        key = f"{db_stem}__{token}"
        if key in already_done:
            continue

        ego = get_ego_pose(conn, token)
        if not ego:
            continue

        image = get_camera_image(conn, data_root, db_stem, token, camera, blob_subdir)
        if image is None:
            skipped_no_image += 1
            continue

        objects = get_tracked_objects(conn, token)
        tl = get_traffic_lights(conn, token)
        ann_text = build_annotation_text(ego, objects, tl)

        try:
            description = generate_fn(image, ann_text)
        except Exception as e:
            description = f"ERROR: {e}"

        record = {
            "key": key,
            "log": db_stem,
            "lidarpc_token": token,
            "ego": ego,
            "objects": objects,
            "traffic_lights": tl,
            "annotation_text": ann_text,
            "description": description,
        }
        output_file.write(json.dumps(record) + "\n")
        output_file.flush()

    if skipped_no_image:
        print(f"  [{db_stem}] skipped {skipped_no_image}/{len(tokens)} frames (image not found)")
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True, type=Path,
                        help="Root of NuPlan data (contains splits/, sensor_blobs_mini/, etc.)")
    parser.add_argument("--split_dir",  default="splits/mini",
                        help="Relative path to folder containing .db files (default: splits/mini)")
    parser.add_argument("--blob_subdir", default="sensor_blobs_mini",
                        help="Subfolder under data_root containing images (default: sensor_blobs_mini)")
    parser.add_argument("--output",     required=True, type=Path)
    parser.add_argument("--model",      default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Model name or 'claude' for Claude API")
    parser.add_argument("--camera",     default="CAM_F0")
    parser.add_argument("--stride",     type=int, default=10,
                        help="Sample 1 frame every N lidarpc frames (default: 10 ≈ 2 Hz at 20 Hz)")
    parser.add_argument("--max_dbs",    type=int, default=None,
                        help="Process only first N DB files (for debugging)")
    parser.add_argument("--shard",      type=int, default=0,
                        help="Shard index for parallel SLURM array jobs")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards")
    args = parser.parse_args()

    # Resume: collect already-done keys
    already_done: set[str] = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["key"])
                except Exception:
                    pass
        print(f"Resuming — {len(already_done)} frames already done.")

    db_files = find_db_files(args.data_root, args.split_dir)
    if not db_files:
        raise FileNotFoundError(f"No .db files in {args.data_root / args.split_dir}")
    print(f"Found {len(db_files)} DB files in {args.split_dir}")

    if args.max_dbs:
        db_files = db_files[: args.max_dbs]
    db_files = db_files[args.shard :: args.num_shards]
    print(f"This shard: {len(db_files)} DB files")

    print(f"Loading model: {args.model} ...")
    _, _, generate_fn = load_model(args.model)
    print("Model ready.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as out:
        for db_path in tqdm(db_files, desc="Logs"):
            process_db(
                db_path, args.data_root, args.blob_subdir,
                generate_fn, args.camera, args.stride, out, already_done,
            )

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
