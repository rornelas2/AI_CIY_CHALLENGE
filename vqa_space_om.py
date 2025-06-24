import os
import json
from pathlib import Path
from tqdm import tqdm

# Config - adjust paths as needed
VQA_ROOT = Path("data/annotations/vqa/val")       # Your VQA JSON annotation root
BBOX_ROOT = Path("data/bbox_global/val")          # Root where images are stored
OUTPUT_JSON = Path("vqa_spaceom_val_multiframe.json")
MAX_FRAMES = 10

label_map = {
    "avoidance": "avoidance",
    "action": "action",
    "judgement": "judgement",
    "recognition": "recognition",
    "prerecognition": "prerecognition",
    "4": "4",  # fallback label
}

def load_json(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def make_content(img_paths, question, choices):
    """
    Build content list: multiple images + question text
    """
    content = [{"type": "image", "image": str(p.relative_to(BBOX_ROOT))} for p in img_paths]
    question_full = question + "\n" + "\n".join(f"{k}: {v}" for k, v in choices.items())
    content.append({"type": "text", "text": question_full})
    return content

def collect_imgs(folder: Path, segment: str):
    """
    Collect up to MAX_FRAMES images from `folder` whose filenames include `segment` (case-insensitive).
    If none found, fallback to first MAX_FRAMES images in the folder.
    """
    if not folder or not folder.exists():
        return []

    segment_lower = segment.lower()
    imgs = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".jpg" and segment_lower in f.name.lower()
    ])
    if not imgs:
        # fallback: any images in folder
        imgs = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
    return imgs[:MAX_FRAMES]

def build_env(sid, env_data):
    out = []
    env_folder = BBOX_ROOT / sid / "environment"

    if not env_folder.exists():
        ov = BBOX_ROOT / sid / "overhead_view"
        if ov.exists():
            env_folder = next((c for c in ov.iterdir() if c.is_dir()), None)
        else:
            env_folder = None

    for block in env_data:
        for q in block.get("environment", []):
            imgs = []
            if env_folder:
                imgs = collect_imgs(env_folder, segment="")  # no segment filtering for env

            if not imgs:
                # Skip question if no images
                # print(f"Warning: No images found for scenario {sid} in environment.")
                continue

            content = make_content(imgs, q["question"], {k: q[k] for k in ('a','b','c','d') if k in q})
            out.append({
                "id": sid,
                "segment": "unknown",
                "view": "environment",
                "start_time": "0",
                "end_time": "0",
                "conversations": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": q.get("correct", "")}]},
                ],
                "image": str(imgs[0].relative_to(BBOX_ROOT))
            })
    return out

def build_overhead(sid, over_data):
    out = []
    entry = over_data[0]
    video_stems = [Path(v).stem for v in entry.get("overhead_videos", [])]
    cams_root = BBOX_ROOT / sid / "overhead_view"

    for phase in entry.get("event_phase", []):
        seg_raw = phase.get("labels", ["unknown"])[0]
        segment = label_map.get(seg_raw, seg_raw).lower()
        start_time = phase.get("start_time", "0")
        end_time = phase.get("end_time", "0")

        imgs = []
        for vs in video_stems:
            cam_folder = cams_root / vs
            if not cam_folder.exists():
                continue

            cand = sorted([
                p for p in cam_folder.iterdir()
                if p.is_file() and p.suffix.lower() == ".jpg" and segment in p.name.lower()
            ])
            imgs.extend(cand)
            if len(imgs) >= MAX_FRAMES:
                break

        imgs = imgs[:MAX_FRAMES]

        # fallback: try any images if none matched segment
        if not imgs:
            for vs in video_stems:
                cam_folder = cams_root / vs
                if not cam_folder.exists():
                    continue
                any_imgs = sorted([p for p in cam_folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
                if any_imgs:
                    imgs = any_imgs[:MAX_FRAMES]
                    break

        if not imgs:
            # Skip if no images found
            # print(f"Warning: No images found for scenario {sid}, segment '{segment}' in overhead_view.")
            continue

        for conv in phase.get("conversations", []):
            content = make_content(imgs, conv["question"], {k: conv[k] for k in ("a","b","c","d") if k in conv})
            out.append({
                "id": sid,
                "segment": segment,
                "view": "overhead",
                "start_time": start_time,
                "end_time": end_time,
                "conversations": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": conv.get("correct", "")}]}
                ],
                "image": str(imgs[0].relative_to(BBOX_ROOT))
            })
    return out

def build_vehicle(sid, veh_data, bbox_root):
    out = []
    entry = veh_data[0]
    event_phases = entry.get("event_phase", [])
    vehicle_cameras_path = bbox_root / sid / "vehicle_view"

    # Attempt to find the nested folder for vehicle images
    nested_folder = None
    if vehicle_cameras_path.exists():
        # Sometimes nested folders named like <sid>_vehicle_view
        possible_nested = vehicle_cameras_path / f"{sid}_vehicle_view"
        if possible_nested.exists():
            nested_folder = possible_nested
        else:
            # fallback to first directory inside vehicle_view
            nested_folder = next((c for c in vehicle_cameras_path.iterdir() if c.is_dir()), None)

    for phase in event_phases:
        seg_raw = phase.get("labels", ["unknown"])[0]
        segment = label_map.get(seg_raw, seg_raw).lower()
        start_time = phase.get("start_time", "0")
        end_time = phase.get("end_time", "0")

        imgs = []
        # First try nested folder
        if nested_folder:
            imgs = collect_imgs(nested_folder, segment)
        # fallback to vehicle_view root folder
        if not imgs:
            imgs = collect_imgs(vehicle_cameras_path, segment)

        # DEBUG: print how many images found
        # print(f"Vehicle view images found for {sid} segment '{segment}': {len(imgs)}")

        if not imgs:
            # Skip if no images found
            # print(f"Warning: No images found for scenario {sid}, segment '{segment}' in vehicle_view.")
            continue

        for conv in phase.get("conversations", []):
            content = make_content(imgs, conv["question"], {k: conv[k] for k in ("a","b","c","d") if k in conv})
            out.append({
                "id": sid,
                "segment": segment,
                "view": "vehicle",
                "start_time": start_time,
                "end_time": end_time,
                "conversations": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": conv.get("correct", "")}]}
                ],
                "image": str(imgs[0].relative_to(BBOX_ROOT))
            })
    return out



def process_scenario(folder: Path, sid: str, sink: list):
    env = load_json(folder / "environment" / f"{sid}.json")
    over = load_json(folder / "overhead_view" / f"{sid}.json")
    veh = load_json(folder / "vehicle_view" / f"{sid}.json")

    if env:
        sink.extend(build_env(sid, env))
    if over:
        sink.extend(build_overhead(sid, over))
    if veh:
        sink.extend(build_vehicle(sid, veh, BBOX_ROOT))

def main():
    all_samples = []

    for scen in tqdm(sorted(os.listdir(VQA_ROOT))):
        scen_path = VQA_ROOT / scen
        if scen_path.is_dir():
            process_scenario(scen_path, scen, all_samples)

    print(f"Total samples created: {len(all_samples)}")

    with open(OUTPUT_JSON, "w") as f_out:
        json.dump(all_samples, f_out, indent=2)

if __name__ == "__main__":
    main()
