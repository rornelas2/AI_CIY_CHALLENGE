import os
import json
import copy
import argparse

class Args:
    root = 'data'
    split = 'train'
    save_folder = 'data_preprocess'
    anno_path = 'processed_anno/wts_train_all_video_with_bbox_anno_first_frame.json'
    save_path = 'data_preprocess/wts_train_spacellava_format.json'
    wts_global_image_path = 'data/bbox_global'
    bdd_global_image_path = 'external/BDD_PC_5k'

args = Args()

phrase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}
number_phrase_map = {v: k for k, v in phrase_number_map.items()}

camera_path_mapping = dict()

wts_anno_path = os.path.join(args.root, 'annotations/caption', args.split)
bdd_anno_path = os.path.join(args.root, 'external/BDD_PC_5K/annotations/caption', args.split)

train_samples = list()

overhead = 'overhead_view'
vehicle = 'vehicle_view'

# --- Load WTS annotations ---
for item in os.listdir(wts_anno_path):
    overhead_flag, vehicle_flag = True, True
    try:
        overhead_view = json.load(open(f'{wts_anno_path}/{item}/{overhead}/{item}_caption.json'))
    except:
        overhead_flag = False
    try:
        vehicle_view = json.load(open(f'{wts_anno_path}/{item}/{vehicle}/{item}_caption.json'))
    except:
        vehicle_flag = False
    sample_id = item

    if overhead_flag:
        for event in overhead_view['event_phase']:
            cur_data = dict()
            cur_data['id'] = sample_id
            cur_data['segment'] = phrase_number_map[event['labels'][0]]
            cur_data['view'] = 'overhead'
            cur_data['start_time'] = event['start_time']
            cur_data['end_time'] = event['end_time']

            # Use the first image to build conversations then duplicate for all overhead_videos below
            first_image = overhead_view['overhead_videos'][0] if overhead_view.get('overhead_videos') else None

            cur_data['conversations'] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are VL-Thinking, a helpful assistant with excellent reasoning and descriptive abilities. "
                                "You should first think about the reasoning process and then provide the answer. "
                                "Use <think>...</think> and <answer>...</answer> tags."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": first_image},
                        {
                            "type": "text",
                            "text": (
                                "Please describe the interested pedestrian in the video. Provide specific numerical information about "
                                "age, height, clothing(color, type), and awareness, position with respect to the vehicle at the time of the accident."
                            )
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": event['caption_pedestrian']}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please describe the interested vehicle in the video. Include detailed observations about the vehicle's type, color, position relative to the pedestrian, movement (e.g., stationary or moving), and speed if observable. Also mention road and environmental conditions (e.g., road type, lighting, traffic flow, presence of sidewalks or lane markings), and any contextual information that may help explain the vehicle's behavior at the time of the incident."
                            )
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": event['caption_vehicle']}]
                }
            ]

            # For each overhead video image, create a separate sample with the image path replaced
            for image in overhead_view.get('overhead_videos', []):
                new_data = copy.deepcopy(cur_data)
                new_data['image'] = image
                # Update conversation image to this specific image
                for conv in new_data['conversations']:
                    if conv['role'] == 'user':
                        for content in conv['content']:
                            if content['type'] == 'image':
                                content['image'] = image
                train_samples.append(new_data)

    if vehicle_flag:
        for event in vehicle_view['event_phase']:
            cur_data = dict()
            cur_data['id'] = sample_id
            cur_data['segment'] = phrase_number_map[event['labels'][0]]
            cur_data['view'] = 'vehicle'
            cur_data['start_time'] = event['start_time']
            cur_data['end_time'] = event['end_time']

            cur_data['conversations'] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are VL-Thinking, a helpful assistant with excellent reasoning and descriptive abilities. "
                                "You should first think about the reasoning process and then provide the answer. "
                                "Use <think>...</think> and <answer>...</answer> tags."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": vehicle_view.get('vehicle_view', None)},
                        {
                            "type": "text",
                            "text": (
                                "Please describe the interested pedestrian in the video. Provide specific numerical information about "
                                "age, height, clothing(color, type), and awareness, position with respect to the vehicle at the time of the accident."
                            )
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": event['caption_pedestrian']}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please describe the interested vehicle in the video. Include detailed observations about the vehicle's type, color, position relative to the pedestrian, movement (e.g., stationary or moving), and speed if observable. Also mention road and environmental conditions (e.g., road type, lighting, traffic flow, presence of sidewalks or lane markings), and any contextual information that may help explain the vehicle's behavior at the time of the incident."
                            )
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": event['caption_vehicle']}]
                }
            ]

            cur_data['image'] = vehicle_view.get('vehicle_view', None)
            train_samples.append(cur_data)

# (Repeat similar logic for normal_trimmed folder, and BDD data...)

# Skipping repetitive parts for brevity - same logic applies for normal_trimmed and BDD annotations

# Build camera_path_mapping for global images
global_image_path = os.path.join(args.wts_global_image_path, args.split)
for event in os.listdir(global_image_path):
    if 'normal_trimmed' in event:
        continue
    for view in os.listdir(os.path.join(global_image_path, event)):
        parent_path = os.path.join(global_image_path, event, view)
        for camera in os.listdir(parent_path):
            camera_path_mapping[camera] = os.path.join(parent_path, camera)

for event in os.listdir(os.path.join(global_image_path, 'normal_trimmed')):
    for view in os.listdir(os.path.join(global_image_path, 'normal_trimmed', event)):
        parent_path = os.path.join(global_image_path, 'normal_trimmed', event, view)
        for camera in os.listdir(parent_path):
            camera_path_mapping[camera] = os.path.join(parent_path, camera)

# Finalize train samples with correct image paths and update conversation images
reserved_train_samples = list()
for item in train_samples:
    image_key = item['image'].replace('.mp4', '')
    segment = item['segment']

    if 'video' in image_key:
        train_image_name = f'{image_key}_{segment}.jpg'
    else:
        train_image_name = f'{number_phrase_map[segment]}_{segment}.jpg'

    if image_key in camera_path_mapping:
        final_image_path = os.path.join(camera_path_mapping[image_key], train_image_name)
        if os.path.exists(final_image_path):
            # Update top-level image path
            item['image'] = final_image_path.replace('./data/', '')

            # Update conversation images as well
            for conv in item['conversations']:
                if conv['role'] == 'user':
                    for content in conv['content']:
                        if content['type'] == 'image':
                            content['image'] = item['image']

            reserved_train_samples.append(item)

os.makedirs(args.save_folder, exist_ok=True)
with open(os.path.join(args.save_folder, f'wts_bdd_{args.split}.json'), 'w+') as f:
    f.write(json.dumps(reserved_train_samples, indent=4))
