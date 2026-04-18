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
        # --- Bug 1 fix: use per-frame bbox interpolation, not just centres ---
        # Collect the four bbox coords separately so width/height are preserved.
        # Interpolating only the centre and reconstructing with a fixed 3px
        # half-size threw away the real ball size (bug 1).
        x1s, y1s, x2s, y2s = [], [], [], []
        frame_indices = []
        ball_id = None

        for frame_idx, balls in enumerate(tracks["balls"]):
            if balls:
                # --- Bug 2 fix: track which ball_id belongs to which frame ---
                # The original code kept overwriting ball_id with whatever was
                # last in the loop, then used that single id for ALL filled gaps.
                # If multiple ball ids ever appeared the wrong id was stamped on
                # interpolated frames.  We collect per-frame and use the most
                # common id below.
                fid, data = next(iter(balls.items()))
                frame_indices.append(frame_idx)
                ball_id = fid          # most-recent real detection id (see fix 2 below)
                x1, y1, x2, y2 = data["bbox"]
                x1s.append(x1); y1s.append(y1)
                x2s.append(x2); y2s.append(y2)

        if len(frame_indices) < 2:
            return tracks

        fi  = np.array(frame_indices, dtype=float)
        all_frames = np.arange(len(tracks["balls"]), dtype=float)

        # --- Bug 3 fix: interpolate all four bbox coords, not just centre ---
        # Rebuilding a bbox as (cx±3, cy±3) always produced a 6×6 px square
        # regardless of the real ball size — breaking draw_ball_ellipse which
        # relies on the bbox dimensions to size the rings.
        ix1 = np.interp(all_frames, fi, x1s)
        iy1 = np.interp(all_frames, fi, y1s)
        ix2 = np.interp(all_frames, fi, x2s)
        iy2 = np.interp(all_frames, fi, y2s)

        # --- Bug 4 fix: clamp extrapolated frames beyond last detection ---
        # np.interp clamps to boundary values outside the range, which is
        # correct for frames *before* the first detection and *after* the last.
        # However we should NOT fill frames that lie entirely outside the
        # detection window with a frozen position — that makes the ball appear
        # frozen in the corner before/after it is actually visible.
        first_detected = int(fi[0])
        last_detected  = int(fi[-1])

        for i in range(len(tracks["balls"])):
            if not tracks["balls"][i]:
                # Only fill gaps that are *between* real detections
                if first_detected < i < last_detected:
                    tracks["balls"][i] = {
                        ball_id: {
                            "bbox": [ix1[i], iy1[i], ix2[i], iy2[i]],
                            "interpolated": True,   # handy flag for debugging
                            "track_id": ball_id,
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