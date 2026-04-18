from ultralytics import YOLO
import supervision as sv
import os
import pickle
import numpy as np
import cv2

from ball_assigners import BallAssigner
from utils import (
    feet_anchor,
    ellipse_axes_from_bbox,
    draw_ground_ellipse,
    draw_id_label,
    draw_inverted_triangle,
    draw_ball_ellipse,
    draw_has_ball_triangle,
    draw_team_has_ball_ellipse,
    draw_possession_hud,
)


class Tracker:
    def __init__(
        self,
        model_path="yolo_models/best.pt",
        batch_size=20,
        no_of_batches=None,
        ball_max_distance=300
    ):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        self.batch_size = batch_size
        self.no_of_batches = no_of_batches

        self.ball_assigner = BallAssigner(max_distance=ball_max_distance)
        self.team_possession_frames = {}

    # --------------------------------------------------
    # DETECTION
    # --------------------------------------------------
    def detect_frames(self, frames):
        detections = []
        for batch_idx in range(0, len(frames), self.batch_size):
            if self.no_of_batches and batch_idx // self.batch_size >= self.no_of_batches:
                break
            batch_frames = frames[batch_idx:batch_idx + self.batch_size]
            results = self.model.predict(batch_frames, verbose=False)
            detections.extend(results)
        return detections

    # --------------------------------------------------
    # BALL INTERPOLATION
    # --------------------------------------------------
    def interpolate_ball_positions(self, tracks):
        ball_centers = []
        frame_indices = []
        ball_id = None

        for frame_idx, balls in enumerate(tracks["balls"]):
            if balls:
                ball_id, data = next(iter(balls.items()))
                bbox = data["bbox"]
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                ball_centers.append(center)
                frame_indices.append(frame_idx)

        if len(frame_indices) < 2:
            return tracks

        frame_indices = np.array(frame_indices)
        ball_centers  = np.array(ball_centers)
        all_frames    = np.arange(len(tracks["balls"]))

        x_interp = np.interp(all_frames, frame_indices, ball_centers[:, 0])
        y_interp = np.interp(all_frames, frame_indices, ball_centers[:, 1])

        for i in range(len(tracks["balls"])):
            if not tracks["balls"][i]:
                tracks["balls"][i] = {
                    ball_id: {
                        "bbox": [
                            x_interp[i] - 3, y_interp[i] - 3,
                            x_interp[i] + 3, y_interp[i] + 3
                        ],
                        "track_id": ball_id
                    }
                }
        return tracks

    # --------------------------------------------------
    # TRACKING
    # --------------------------------------------------
    def get_object_tracks(self, frames, read_backup=True, backup_path="runs/backups/tracks.pkl"):
        if read_backup and os.path.exists(backup_path):
            with open(backup_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "balls": []}

        for frame_idx, detection in enumerate(detections):
            sv_dets    = sv.Detections.from_ultralytics(detection)
            class_names = detection.names
            name_to_id  = {v: k for k, v in class_names.items()}
            player_class_id = name_to_id.get("player")

            for i, class_id in enumerate(sv_dets.class_id):
                if class_names[class_id] == "goalkeeper":
                    sv_dets.class_id[i] = player_class_id
                    sv_dets.data["class_name"][i] = "player"

            tracked_dets = self.tracker.update_with_detections(sv_dets)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["balls"].append({})

            for det in tracked_dets:
                bbox     = det[0].tolist()
                track_id = int(det[4])
                cls_name = det[5]["class_name"]
                data     = {"bbox": bbox, "track_id": track_id}

                if cls_name == "player":
                    tracks["players"][frame_idx][track_id] = data
                elif cls_name == "referee":
                    tracks["referees"][frame_idx][track_id] = data
                elif cls_name == "ball":
                    tracks["balls"][frame_idx][track_id] = data

        tracks = self.interpolate_ball_positions(tracks)

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, "wb") as f:
            pickle.dump(tracks, f)

        return tracks

    # --------------------------------------------------
    # BALL POSSESSION LOGIC
    # --------------------------------------------------
    def update_ball_possession(self, tracks, frame_idx):
        players = tracks["players"][frame_idx]
        balls   = tracks["balls"][frame_idx]

        for data in players.values():
            data.pop("has_ball", None)
            data.pop("team_has_ball", None)

        assigned_player = self.ball_assigner.assign_ball_to_player(players, balls)
        if assigned_player is None:
            return None

        owner_data = players.get(assigned_player)
        if owner_data is None:
            return None

        owner_team = owner_data.get("team")
        if owner_team is None:
            return None

        owner_team = int(owner_team)
        owner_data["has_ball"] = True

        for data in players.values():
            if int(data.get("team", -1)) == owner_team:
                data["team_has_ball"] = True

        self.team_possession_frames.setdefault(owner_team, 0)
        self.team_possession_frames[owner_team] += 1

        return owner_team

    # --------------------------------------------------
    # COLLECT TEAM COLOURS FOR HUD
    # --------------------------------------------------
    def _collect_team_colors(self, players_frame):
        """
        Scans the current frame's player data and builds a
        {team_id: BGR_color} dict for the HUD.
        """
        colors = {}
        for data in players_frame.values():
            team = data.get("team")
            col  = data.get("team_color")
            if team is not None and col is not None:
                colors[int(team)] = tuple(int(c) for c in col)
        return colors if colors else None

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    def visualize_tracks(self, frames, tracks, camera_movements=None, max_frames=None):
        visualized_frames = []

        for frame_idx, frame in enumerate(frames):
            if max_frames and frame_idx >= max_frames:
                break

            frame  = frame.copy()
            frame_h = frame.shape[0]

            controlling_team = self.update_ball_possession(tracks, frame_idx)

            # -------- Camera motion arrow --------
            if camera_movements and frame_idx in camera_movements:
                cx, cy = camera_movements[frame_idx]
                h, w   = frame.shape[:2]
                center = (w // 2, h // 2)
                scale  = 6
                end_point = (int(center[0] + cx * scale), int(center[1] + cy * scale))
                cv2.arrowedLine(frame, center, end_point, (200, 200, 255), 2, tipLength=0.3)
                label = f"cam  dx={cx:.1f}  dy={cy:.1f}"
                cv2.putText(frame, label, (w // 2 - 80, h // 2 - 14),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0),       2, cv2.LINE_AA)
                cv2.putText(frame, label, (w // 2 - 80, h // 2 - 14),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)

            # -------- Players --------
            for track_id, data in tracks["players"][frame_idx].items():
                bbox   = data["bbox"]
                center = feet_anchor(bbox)
                axes   = ellipse_axes_from_bbox(bbox, frame_h)
                color  = data.get("team_color", (60, 60, 220))

                # team possession ring (drawn first, behind everything)
                if data.get("team_has_ball"):
                    draw_team_has_ball_ellipse(frame, center, axes)

                # team-coloured shadow disc
                draw_ground_ellipse(frame, center, axes, color)

                # pill ID badge
                draw_id_label(frame, str(track_id), center, bg_color=color)

                # ball-carrier arrow above head
                if data.get("has_ball"):
                    draw_has_ball_triangle(frame, bbox)

            # -------- Referees --------
            for track_id, data in tracks["referees"][frame_idx].items():
                bbox   = data["bbox"]
                center = feet_anchor(bbox)
                axes   = ellipse_axes_from_bbox(bbox, frame_h)
                draw_ground_ellipse(frame, center, axes, (30, 30, 30))
                draw_id_label(frame, f"R{track_id}", center, bg_color=(30, 30, 30))

            # -------- Ball --------
            for _, data in tracks["balls"][frame_idx].items():
                draw_ball_ellipse(frame, data["bbox"])

            # -------- Possession HUD --------
            team_colors = self._collect_team_colors(tracks["players"][frame_idx])
            draw_possession_hud(frame, self.team_possession_frames, team_colors)

            visualized_frames.append(frame)

        return visualized_frames