import cv2
import os
from pathlib import Path

# Paths (adjust these)
VAL_VIDEO_ROOT = Path("/home/rornelas/Desktop/Mi3_Lab/AI_CIY_CHALLENGE/data/videos/val")  # path to validation videos
OUTPUT_ROOT = Path("/home/rornelas/Desktop/Mi3_Lab/AI_CIY_CHALLENGE/data/bbox_global/val")  # where to save extracted frames

def extract_frames_from_video(video_path, output_folder, frame_interval=500):
    """
    Extract frames every `frame_interval` frames and save to output_folder.
    frame_interval=30 means approx 1 frame per second if video fps is ~30.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame as jpg
            frame_filename = output_folder / f"{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

def main():
    for root, dirs, files in os.walk(VAL_VIDEO_ROOT):
        for file in files:
            if file.endswith(".mp4"):
                video_path = Path(root) / file
                # Build output path mirroring the video folder structure
                relative_path = video_path.relative_to(VAL_VIDEO_ROOT)
                output_folder = OUTPUT_ROOT / relative_path.parent / relative_path.stem
                print(f"Processing video: {video_path}")
                extract_frames_from_video(video_path, output_folder, frame_interval=150)

if __name__ == "__main__":
    main()
