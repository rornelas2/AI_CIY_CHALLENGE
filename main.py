import json
import os
from utils import process_all_json_files_recursive, extract_scenario_id


def main():
    root_foler = "data/annotations/caption"
    output_file = "outputs/submission_dummy_captions.json"
    data = process_all_json_files_recursive(root_foler)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent = 4)

if __name__ == '__main__':
    main()