import logging
import argparse

from utils import read_video, save_video
from trackers import Tracker
from team_assigners import TeamAssigner
from camera_movement import CameraMovement, apply_camera_compensation
from perspective_transformer import PerspectiveTransformer
from speed_and_distance import SpeedAndDistance 

# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "video_samples/08fd33_4.mp4"
SAVE_PATH = "runs/outputs/result.mp4"
LOG_LEVEL = logging.INFO

# Court dimensions (world space, meters)
COURT_WIDTH = 68
COURT_LENGTH = 23.32

# Court corners in image space (pixel coordinates)
PIXEL_VERTICES = [
    [110, 1035],   # bottom-left
    [265, 275],    # top-left
    [910, 260],    # top-right
    [1640, 915],   # bottom-right
]

# ============================================================
# ARGUMENT PARSER (ONLY MODEL PATH)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo_models/best.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="runs/outputs/result.mp4",
        help="Path to save output video"
    )
    return parser.parse_args()

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    args = parse_args()

    # -----------------------------
    # Logging setup
    # -----------------------------
    logging.basicConfig(
        level=LOG_LEVEL,
        format="[%(levelname)s] %(message)s"
    )

    logging.info("Starting video processing pipeline")

    # -----------------------------
    # Read video
    # -----------------------------
    frames = read_video(video_path=VIDEO_PATH)
    if not frames:
        logging.error("No frames loaded. Check video path.")
        return

    logging.info(f"Loaded {len(frames)} frames")

    # -----------------------------
    # Detection + Tracking
    # -----------------------------
    tracker = Tracker(model_path=args.model_path)

    tracks = tracker.get_object_tracks(
        frames,
        read_backup=True
    )

    if "players" not in tracks:
        logging.error("No player tracks found.")
        return

    logging.info("Tracking completed")

    # -----------------------------
    # Team assignment initialization
    # -----------------------------
    team_assigner = TeamAssigner(refresh_interval=30)

    if tracks["players"] and tracks["players"][0]:
        team_assigner.assign_team_color(
            frames[0],
            tracks["players"][0]
        )
        logging.info("Team colors initialized")
    else:
        logging.warning("No players detected in first frame. Skipping team init.")

    # -----------------------------
    # Camera Movement Estimation
    # -----------------------------
    logging.info("Estimating camera movement...")
    cam_estimator = CameraMovement()
    camera_movements = cam_estimator.estimate(frames, visualize=False)
    logging.info("Camera movement estimation completed")

    # -----------------------------
    # Apply camera compensation
    # -----------------------------
    tracks = apply_camera_compensation(tracks, camera_movements)
    logging.info("Applied camera compensation to tracks")

    print(tracks["players"][0])

    # -----------------------------
    # Perspective Transformation
    # -----------------------------
    perspective_transformer = PerspectiveTransformer(
        pixel_vertices=PIXEL_VERTICES,
        court_length=COURT_LENGTH,
        court_width=COURT_WIDTH,
    )
    logging.info("Perspective transformer initialized")

    # -----------------------------
    # Per-frame processing
    # -----------------------------
    for frame_idx, frame in enumerate(frames):
        team_assigner.step()

        players_in_frame = tracks["players"][frame_idx]

        for track_id, player_data in players_in_frame.items():

            # Team assignment
            team_id = team_assigner.get_player_team(frame, player_data)
            if team_id is not None:
                player_data["team"] = team_id
                player_data["team_color"] = team_assigner.team_colors[team_id]

            # Perspective transform
            if "adjusted_position" in player_data:
                x, y = player_data["adjusted_position"]
                world_pos = perspective_transformer.transform_point(x, y)
                player_data["transformed_position"] = world_pos
            else:
                player_data["transformed_position"] = None

        # Ball possession
        tracker.update_ball_possession(tracks, frame_idx)

        if frame_idx % 50 == 0:
            logging.info(f"Processed frame {frame_idx}/{len(frames)}")

    # -----------------------------
    # Speed & Distance computation
    # -----------------------------
    logging.info("Computing player speed and distance")
    speed_distance_estimator = SpeedAndDistance()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # -----------------------------
    # Visualization
    # -----------------------------
    logging.info("Rendering tracks, camera movement, speed and distance")
    vis_frames = tracker.visualize_tracks(frames, tracks)
    vis_frames = cam_estimator.draw_camera_movement(vis_frames, camera_movements)
    vis_frames = speed_distance_estimator.draw_speed_and_distance(vis_frames, tracks)

    # -----------------------------
    # Save output
    # -----------------------------
    save_video(vis_frames, args.output_path, fps=30)
    logging.info(f"Result saved to {args.output_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()