import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utility
# ============================================================

def bbox_center_xyxy(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


# ============================================================
# Camera Movement Estimator
# ============================================================

class CameraMovement:
    def __init__(self):

        # -------------------------
        # Feature parameters
        # -------------------------
        self.max_corners = 50
        self.quality_level = 0.01
        self.min_distance = 10

        # Important area (ROI)
        self.object_mask_coord = np.array([
            [10, 10],
            [10, 190],
            [1909, 190],
            [1909, 10]
        ], dtype=np.int32)

        # Lucas–Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Thresholds
        self.min_dist = 5
        self.max_dist = 10

    # --------------------------------------------------
    # Estimate camera movement per frame
    # --------------------------------------------------
    def estimate(self, frames, visualize=False):

        h, w = frames[0].shape[:2]
        object_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(object_mask, [self.object_mask_coord], 255)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=object_mask
        ).astype(np.float32)

        camera_movements = {0: (0.0, 0.0)}

        for frame_num in range(1, len(frames)):

            max_distance = 0.0
            cx, cy = 0.0, 0.0

            frame = frames[frame_num].copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_points,
                None,
                **self.lk_params
            )

            if next_points is None:
                prev_gray = gray
                continue

            good_old = prev_points[status == 1]
            good_new = next_points[status == 1]

            if len(good_new) < 5:
                prev_gray = gray
                continue

            # ---- Find strongest motion ----
            for old, new in zip(good_old, good_new):
                dx = new[0] - old[0]
                dy = new[1] - old[1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist > max_distance:
                    max_distance = dist
                    cx, cy = dx, dy

            camera_movements[frame_num] = (cx, cy)

            # ---- Visualization ----
            if visualize:
                self._visualize(frame, cx, cy, max_distance, frame_num)

            # ---- Re-detect if needed ----
            if max_distance > self.max_dist:
                prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    mask=object_mask
                ).astype(np.float32)
            else:
                prev_points = good_new.reshape(-1, 1, 2)

            prev_gray = gray

        return camera_movements

    # --------------------------------------------------
    # Visualization helper
    # --------------------------------------------------
    def _visualize(self, frame, cx, cy, max_dist, frame_num):

        vis = frame.copy()

        if max_dist < self.min_dist:
            text, color = "NO CAMERA MOVE", (255, 255, 0)
        elif max_dist > self.max_dist:
            text, color = "HUGE MOVE", (0, 0, 255)
        else:
            text, color = "CAMERA MOVING", (0, 255, 0)

        cv2.putText(vis, f"Frame: {frame_num}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(vis, f"cx={cx:.2f}, cy={cy:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.putText(vis, text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("Camera Movement", vis)
        cv2.waitKey(30)

    # --------------------------------------------------
    # Draw movement on frames
    # --------------------------------------------------
    def draw_camera_movement(self, frames, camera_movements):

        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (50, 50), (520, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cx, cy = camera_movements.get(frame_num, (0, 0))

            cv2.putText(frame, f"Camera X: {cx:.2f}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Y: {cy:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames


# ============================================================
# Apply camera compensation to tracked objects
# ============================================================

def apply_camera_compensation(tracks, camera_movements):

    for obj_id, track_objects in tracks.items():
        for frame_num, frame_objects in enumerate(track_objects):

            cam_x, cam_y = camera_movements.get(frame_num, (0, 0))

            for track_id, obj in frame_objects.items():
                cx, cy = bbox_center_xyxy(obj["bbox"])

                obj["position"] = (cx, cy)

                adj_cx = cx - cam_x
                adj_cy = cy - cam_y

                obj["adjusted_position"] = (adj_cx, adj_cy)

    return tracks
