import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=30):
    if len(output_video_frames) == 0:
        raise ValueError("No frames to save")

    # Create parent directory if needed
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    height, width, _ = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    for frame in output_video_frames:
        out.write(frame)

    out.release()
