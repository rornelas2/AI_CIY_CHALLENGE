import os
import json
from generate_dummy_captions import generate_dummy_captions

def process_all_json_files_recursive(root_folder):
    
    scenarios = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        print(f"Checking directory: {dirpath}")
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    scenario_id = extract_scenario_id(file_path)
                    
                    for event in data.get("event_phase", []):
                        dummy = generate_dummy_captions(event)
                        if scenario_id not in scenarios:
                            scenarios[scenario_id] = []
                        scenarios[scenario_id].append(dummy)
    
                except Exception as e:
                    print(f"failed to process {file_path}: {e}")
    return scenarios
    

def extract_scenario_id(file_path):
    parts = file_path.split(os.sep)
    
    if len(parts) >= 3:
        return parts[-3]
    return "unknown_scenario"