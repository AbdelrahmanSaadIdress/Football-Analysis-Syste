import numpy as np
import cv2


class PerspectiveTransformer:
    def __init__(self, pixel_vertices, court_length, court_width):
        """
        pixel_vertices : (4, 2) array of court corners in image space
        court_length   : real-world court length (meters)
        court_width    : real-world court width (meters)
        """

        self.court_length = float(court_length)
        self.court_width = float(court_width)

        # Image-space court polygon
        self.pixel_vertices = np.asarray(pixel_vertices, dtype=np.float32)

        # Target (world) coordinates
        self.target_vertices = np.array(
            [
                [0, self.court_width],
                [0, 0],
                [self.court_length, 0],
                [self.court_length, self.court_width],
            ],
            dtype=np.float32,
        )

        # Homography
        self.H = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, x, y):
        """
        Transform a single image-space point to world coordinates.
        Returns (tx, ty) or None if invalid.
        """

        # --------------------------------------------------
        # 1. Image-space validity check
        # --------------------------------------------------
        p = (int(x), int(y))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
            return None

        # --------------------------------------------------
        # 2. Perspective transform
        # --------------------------------------------------
        pts = np.array([[[x, y]]], dtype=np.float32)
        tx, ty = cv2.perspectiveTransform(pts, self.H).ravel()

        # --------------------------------------------------
        # 3. World-space sanity check
        # --------------------------------------------------
        if 0 <= tx <= self.court_length and 0 <= ty <= self.court_width:
            return float(tx), float(ty)

        return None
