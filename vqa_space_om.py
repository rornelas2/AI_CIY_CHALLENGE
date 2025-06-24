import os
import json
from pathlib import Path
from tqdm import tqdm

# Paths - adjust these as needed
VQA_ROOT = Path("data/annotations/vqa/val")
BBOX_ROOT = Path("data/bbox_global/val")
BEST_CAMERA_PATH = Path("processed_anno/best_view_for_scenario.json")
OUTPUT_PATH = Path("vqa_spaceom_val.json")

# Load best camera mapping once
with open(BEST_CAMERA_PATH, "r") as f:
    best_views = json.load(f)

label_map = {
    "avoidance": "avoidance",
    "action": "action",
    "judgement": "judgement",
    "recognition": "recognition",
    "prerecognition": "prerecognition",
    "4": "4",  # default fallback
}

def build_environment_samples(scenario_id, json_data, best_views, bbox_root):
    samples = []
    image_path = ""

   
        
    overhead_cameras_path = bbox_root / scenario_id / "overhead_view"
    image_path = None
    if overhead_cameras_path.exists():
        for cam_folder in overhead_cameras_path.iterdir():
            candidate_image = cam_folder / "00001.jpg"
            image_path = str(candidate_image)
        

    camera_folder = bbox_root / scenario_id / "overhead_view"
    print(f"Looking in folder: {camera_folder}")
    print(f"Folder exists? {camera_folder.exists()}")
    if camera_folder.exists():
        img_files = sorted([f for f in camera_folder.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        print(f"Found {len(img_files)} images:")
        for img in img_files:
            print(img)
        if img_files:
            image_path = str(img_files[0])  # use first image

    for item in json_data:
        questions = item.get("environment", [])
        for q in questions:
            question_text = q.get("question", "")
            choices = {k: q[k] for k in ['a', 'b', 'c', 'd'] if k in q}
            correct_letter = q.get("correct", "Unknown")
            sample = {
                "id": scenario_id,
                "segment": "unknown",
                "view": "environment",
                "start_time": "0",
                "end_time": "0",
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {
                                "type": "text",
                                "text": question_text + "\n" + "\n".join(f"{k}: {v}" for k, v in choices.items())
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": correct_letter}
                        ]
                    }
                ],
                "image": image_path
            }
            samples.append(sample)
    return samples

def build_overhead_samples(scenario_id, json_data, bbox_root):
    samples = []
    if not json_data or not isinstance(json_data, list):
        return samples

    event_phases = json_data[0].get("event_phase", [])
    for phase in event_phases:
        start_time = phase.get("start_time", "0")
        end_time = phase.get("end_time", "0")
        labels = phase.get("labels", ["4"])
        segment_label = labels[0]
        segment = label_map.get(segment_label, segment_label)

        overhead_cameras_path = bbox_root / scenario_id / "overhead_view"
        image_path = None
        if overhead_cameras_path.exists():
            for cam_folder in overhead_cameras_path.iterdir():
                candidate_image = cam_folder / f"{segment}.jpg"
                image_path = str(candidate_image)
        

        for conv in phase.get("conversations", []):
            question_text = conv.get("question", "")
            choices = {k: conv[k] for k in ['a', 'b', 'c', 'd'] if k in conv}
            correct_letter = conv.get("correct", "Unknown")

            sample = {
                "id": scenario_id,
                "segment": segment,
                "view": "overhead",
                "start_time": start_time,
                "end_time": end_time,
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path or ""},
                            {
                                "type": "text",
                                "text": question_text + "\n" + "\n".join(f"{k}: {v}" for k, v in choices.items())
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": correct_letter}
                        ]
                    }
                ],
                "image": image_path or ""
            }
            samples.append(sample)
    return samples

def build_vehicle_samples(scenario_id, json_data, bbox_root):
    samples = []
    if not json_data or not isinstance(json_data, list):
        return samples

    video_filename = json_data[0].get("vehicle_view", "")
    event_phases = json_data[0].get("event_phase", [])
    vehicle_cameras_path = bbox_root / scenario_id / "vehicle_view"

    for phase in event_phases:
        start_time = phase.get("start_time", "0")
        end_time = phase.get("end_time", "0")
        labels = phase.get("labels", ["4"])
        segment_label = labels[0]
        segment = label_map.get(segment_label, segment_label)

        image_path = None
        if vehicle_cameras_path.exists() and video_filename:
            candidate_image = vehicle_cameras_path / f"{segment}.jpg"
            image_path = str(candidate_image)

        for conv in phase.get("conversations", []):
            question_text = conv.get("question", "")
            choices = {k: conv[k] for k in ['a', 'b', 'c', 'd'] if k in conv}
            correct_letter = conv.get("correct", "Unknown")

            sample = {
                "id": scenario_id,
                "segment": segment,
                "view": "vehicle",
                "start_time": start_time,
                "end_time": end_time,
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path or ""},
                            {
                                "type": "text",
                                "text": question_text + "\n" + "\n".join(f"{k}: {v}" for k, v in choices.items())
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": correct_letter}
                        ]
                    }
                ],
                "image": image_path or ""
            }
            samples.append(sample)
    return samples

def main():
    train_samples = []

    for scenario_folder in tqdm(sorted(os.listdir(VQA_ROOT))):
        scenario_path = VQA_ROOT / scenario_folder
        if not scenario_path.is_dir():
            continue

        # Handle normal_trimmed differently since folder structure is nested
        if scenario_folder == "normal_trimmed":
            # Iterate subfolders inside normal_trimmed
            for subfolder in sorted(scenario_path.iterdir()):
                if not subfolder.is_dir():
                    continue
                subfolder_name = subfolder.name
                env_json_path = subfolder / "environment" / f"{subfolder_name}.json"
                overhead_json_path = subfolder / "overhead_view" / f"{subfolder_name}.json"
                vehicle_json_path = subfolder / "vehicle_view" / f"{subfolder_name}.json"

                env_data = overhead_data = vehicle_data = None

                if env_json_path.exists():
                    with open(env_json_path, "r") as f:
                        env_data = json.load(f)

                if overhead_json_path.exists():
                    with open(overhead_json_path, "r") as f:
                        overhead_data = json.load(f)

                if vehicle_json_path.exists():
                    with open(vehicle_json_path, "r") as f:
                        vehicle_data = json.load(f)

                print(f"Processing scenario: {subfolder_name}")

                if env_data:
                    print(f"  Environment view, questions: {sum(len(item.get('environment', [])) for item in env_data)}")
                    env_samples = build_environment_samples(subfolder_name, env_data, best_views, BBOX_ROOT)
                    train_samples.extend(env_samples)
                else:
                    print("  Skipping environment - JSON missing")

                if overhead_data:
                    print(f"  Overhead view, event phases: {len(overhead_data[0].get('event_phase', []))}")
                    overhead_samples = build_overhead_samples(subfolder_name, overhead_data, BBOX_ROOT)
                    train_samples.extend(overhead_samples)
                else:
                    print("  Skipping overhead_view - JSON missing")

                if vehicle_data:
                    print(f"  Vehicle view, event phases: {len(vehicle_data[0].get('event_phase', []))}")
                    vehicle_samples = build_vehicle_samples(subfolder_name, vehicle_data, BBOX_ROOT)
                    train_samples.extend(vehicle_samples)
                else:
                    print("  Skipping vehicle_view - JSON missing")

        else:
            # Regular scenario folder
            env_json_path = scenario_path / "environment" / f"{scenario_folder}.json"
            overhead_json_path = scenario_path / "overhead_view" / f"{scenario_folder}.json"
            vehicle_json_path = scenario_path / "vehicle_view" / f"{scenario_folder}.json"

            env_data = overhead_data = vehicle_data = None

            if env_json_path.exists():
                with open(env_json_path, "r") as f:
                    env_data = json.load(f)

            if overhead_json_path.exists():
                with open(overhead_json_path, "r") as f:
                    overhead_data = json.load(f)

            if vehicle_json_path.exists():
                with open(vehicle_json_path, "r") as f:
                    vehicle_data = json.load(f)

            print(f"Processing scenario: {scenario_folder}")

            if env_data:
                print(f"  Environment view, questions: {sum(len(item.get('environment', [])) for item in env_data)}")
                env_samples = build_environment_samples(scenario_folder, env_data, best_views, BBOX_ROOT)
                train_samples.extend(env_samples)
            else:
                print("  Skipping environment - JSON missing")

            if overhead_data:
                print(f"  Overhead view, event phases: {len(overhead_data[0].get('event_phase', []))}")
                overhead_samples = build_overhead_samples(scenario_folder, overhead_data, BBOX_ROOT)
                train_samples.extend(overhead_samples)
            else:
                print("  Skipping overhead_view - JSON missing")

            if vehicle_data:
                print(f"  Vehicle view, event phases: {len(vehicle_data[0].get('event_phase', []))}")
                vehicle_samples = build_vehicle_samples(scenario_folder, vehicle_data, BBOX_ROOT)
                train_samples.extend(vehicle_samples)
            else:
                print("  Skipping vehicle_view - JSON missing")

    print(f"Converted {len(train_samples)} VQA samples.")

    with open(OUTPUT_PATH, "w") as out_f:
        json.dump(train_samples, out_f, indent=2)

if __name__ == "__main__":
    main()
