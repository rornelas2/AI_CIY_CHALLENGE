import json
import os

def generate_dummy_captions(data):
    labels = data.get("labels", [])
    return {
        "labels": labels,
        "caption_pedestrian": "Hello World",
        "caption_vehicle": "Hello World"
    }


def save_output(output, output_path = "output/dummy_output.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved {len(output)} items to output {output_path}")