from decord import VideoReader
import cv2
import numpy as np
import json 
import os
from tqdm import tqdm
from multiprocessing import Pool
import copy

# Hardcoded parameters
ANNO_PATH = 'processed_anno/wts_train_all_video_with_bbox_anno_first_frame.json'
NUM_PROCESSES = 4
SCALE = 1.5

phase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}


def extract_frames(video_path, frame_indices, original_frame_indices):
    vr = VideoReader(video_path)
    if frame_indices[-1] == len(vr):
        frame_indices[-1] = len(vr) - 1
    frames = {ori_idx: vr[frame_idx].asnumpy() for frame_idx, ori_idx in zip(frame_indices, original_frame_indices)}
    return frames


def draw_and_save_bboxes(key, frames, ped_bboxes, veh_bboxes, phase_numbers, phase_number_map):
    for frame_id, frame_np in frames.items():
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        if str(frame_id) in ped_bboxes:
            bbox = ped_bboxes[str(frame_id)]
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=4)
        if str(frame_id) in veh_bboxes:
            bbox = veh_bboxes[str(frame_id)]
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0), thickness=4)

        phase_number = phase_numbers.get(str(frame_id), "")
        if str(phase_number):
            if 'BDD' in key:
                file_name = key.replace('.mp4', f'_{phase_number_map[str(phase_number)]}.jpg').replace('/videos', '/bbox_global')
                dirname = os.path.dirname(file_name)
                os.makedirs(dirname, exist_ok=True)
            else:
                key = key.replace('.mp4', '/').replace('/videos', '/bbox_global')
                os.makedirs(key, exist_ok=True)
                file_name = f"{key}{phase_number}_{phase_number_map[str(phase_number)]}.jpg"

            cv2.imwrite(file_name, frame)


def enlarge_bbox(bbox, scale=1.2):
    xmin, ymin, width, height = bbox
    center_x, center_y = xmin + width / 2, ymin + height / 2
    new_width = width * scale
    new_height = height * scale
    new_xmin = center_x - new_width / 2
    new_ymin = center_y - new_height / 2
    return new_xmin, new_ymin, new_width, new_height


def enlarge_bbox_square(bbox, scale=1.2):
    xmin, ymin, width, height = bbox
    center_x, center_y = xmin + width / 2, ymin + height / 2
    new_width = width * scale
    new_height = height * scale
    new_height, new_width = max(new_width, new_height), max(new_width, new_height)
    new_xmin = center_x - new_width / 2
    new_ymin = center_y - new_height / 2
    return new_xmin, new_ymin, new_width, new_height


def calculate_combined_bbox(bbox1, bbox2):
    xmin = min(bbox1[0], bbox2[0])
    ymin = min(bbox1[1], bbox2[1])
    xmax = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    ymax = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    return xmin, ymin, xmax - xmin, ymax - ymin


def constrain_bbox_within_frame(bbox, frame_shape):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(frame_shape[1], int(xmax))
    ymax = min(frame_shape[0], int(ymax))
    return xmin, ymin, xmax, ymax


def draw_and_save_bboxes_scale_version(key, frames, ped_bboxes, veh_bboxes, phase_numbers, phase_number_map, scale=1.5):
    for frame_id, frame_np in frames.items():
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        combined_bbox = None

        if str(frame_id) in ped_bboxes:
            bbox = enlarge_bbox(ped_bboxes[str(frame_id)], scale)
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            xmin, ymin, xmax, ymax = constrain_bbox_within_frame((xmin, ymin, xmax, ymax), frame.shape)
            combined_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)

        if str(frame_id) in veh_bboxes:
            bbox = enlarge_bbox(veh_bboxes[str(frame_id)], scale)
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            xmin, ymin, xmax, ymax = constrain_bbox_within_frame((xmin, ymin, xmax, ymax), frame.shape)
            if combined_bbox is not None:
                combined_bbox = calculate_combined_bbox(combined_bbox, (xmin, ymin, xmax - xmin, ymax - ymin))
            else:
                combined_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=3)

        if combined_bbox is not None:
            combined_bbox = enlarge_bbox_square(combined_bbox, scale)
            xmin, ymin, width, height = combined_bbox
            xmax, ymax = int(xmin + width), int(ymin + height)
            xmin, ymin = int(xmin), int(ymin)
            xmin, ymin, xmax, ymax = constrain_bbox_within_frame((xmin, ymin, xmax, ymax), frame.shape)
            cropped_frame = frame[ymin:ymax, xmin:xmax]
        else:
            cropped_frame = frame

        if str(frame_id) in phase_numbers:
            phase_number = phase_numbers[str(frame_id)]
        else:
            phase_number = ''

        if str(phase_number):
            if 'BDD' in key:
                file_name = key.replace('.mp4', f'_{phase_number_map[str(phase_number)]}.jpg').replace('/videos', '/bbox_local')
                dirname = os.path.dirname(file_name)
                os.makedirs(dirname, exist_ok=True)
            else:
                key = key.replace('.mp4', '/').replace('/videos', '/bbox_local')
                os.makedirs(key, exist_ok=True)
                file_name = f"{key}{phase_number}_{phase_number_map[str(phase_number)]}.jpg"

            if cropped_frame.size > 0:
                cv2.imwrite(file_name, cropped_frame)
            else:
                print(f"Empty frame: {file_name}")


def process_video(job_args):
    video_path, data, phase_number_map, scale = job_args
    frame_indices = list(map(int, data["phase_number"].keys()))
    if len(frame_indices) == 0:
        return
    frame_indices_process = copy.deepcopy(frame_indices)
    if 'fps' in data:
        if float(data['fps']) > 40.0:
            for i in range(len(frame_indices)):
                frame_indices_process[i] = frame_indices_process[i] // 2
    frames = extract_frames(video_path, frame_indices_process, frame_indices)
    draw_and_save_bboxes(video_path, frames, data["ped_bboxes"], data["veh_bboxes"], data["phase_number"], phase_number_map)
    draw_and_save_bboxes_scale_version(video_path, frames, data["ped_bboxes"], data["veh_bboxes"], data["phase_number"], phase_number_map, scale)


# ==== MAIN EXECUTION START ====
if __name__ == '__main__':
    anno = json.load(open(ANNO_PATH))

    with Pool(processes=NUM_PROCESSES) as pool:
        jobs = []
        for video_path, data in tqdm(anno.items(), desc="Scheduling jobs"):
            job = (video_path, data, phase_number_map, SCALE)
            jobs.append(job)
        results = list(tqdm(pool.imap(process_video, jobs), total=len(jobs), desc="Processing videos"))
