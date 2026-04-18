import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self, refresh_interval=30):
        self.team_colors = {}              # {team_id: BGR_color}
        self.player_team_dict = {}         # {track_id: team_id}
        self.player_color_cache = {}       # {track_id: BGR_color}
        self.frame_counter = 0
        self.refresh_interval = refresh_interval
        self.kmeans_team = None
        self.goalkeeper_ids =  {91: 1, 118: 1, 129:2, 225:1 }  # Example goalkeeper track IDs for team assignment

    # --------------------------------------------------
    # PLAYER COLOR EXTRACTION (BGR)
    # --------------------------------------------------
    def get_player_color(self, frame, bbox):
        """
        Extract dominant jersey color (BGR) from upper body.
        Returns BGR color or None.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp bbox
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Upper torso only
        crop = crop[: crop.shape[0] // 2]
        if crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3)

        # Downsample for speed
        if pixels.shape[0] > 600:
            idx = np.random.choice(pixels.shape[0], 600, replace=False)
            pixels = pixels[idx]

        # Foreground/background clustering
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_

        # Background usually darker; pick brighter as jersey
        fg_label = np.argmax(np.sum(centers, axis=1))
        return centers[fg_label].astype(int)

    # --------------------------------------------------
    # TEAM COLOR INITIALIZATION
    # --------------------------------------------------
    def assign_team_color(self, frame, player_tracks):
        """
        Cluster player colors into two teams.
        Stores final team colors in BGR (for drawing).
        """
        player_colors = []
        track_ids = []

        for track_id, data in player_tracks.items():
            color = self.get_player_color(frame, data["bbox"])
            if color is not None:
                player_colors.append(color)
                track_ids.append(track_id)
                self.player_color_cache[track_id] = color

        if len(player_colors) < 2:
            raise RuntimeError("Not enough player colors to assign teams")

        self.kmeans_team = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = self.kmeans_team.fit_predict(player_colors)
        centers = self.kmeans_team.cluster_centers_

        # Store as int tuples
        self.team_colors = {i + 1: tuple(map(int, centers[i])) for i in range(2)}

    # --------------------------------------------------
    # TEAM PREDICTION
    # --------------------------------------------------
    def get_player_team(self, frame, player_data):
        """
        Predict team for a single player track.
        """
        track_id = player_data["track_id"]

        if track_id in self.player_team_dict:
            return self.player_team_dict[track_id]

        # --- BRUTE FORCE GOALKEEPER OVERRIDE ---
        if track_id in self.goalkeeper_ids:
            team_id = self.goalkeeper_ids[track_id]
            self.player_team_dict[track_id] = team_id
            return team_id


        color = self.player_color_cache.get(track_id)
        if color is None or self.frame_counter % self.refresh_interval == 0:
            color = self.get_player_color(frame, player_data["bbox"])
            if color is None:
                return None
            self.player_color_cache[track_id] = color

        label = self.kmeans_team.predict(color.reshape(1, -1))[0]
        team_id = label + 1

        self.player_team_dict[track_id] = team_id
        return team_id

    # --------------------------------------------------
    # FRAME STEP
    # --------------------------------------------------
    def step(self):
        self.frame_counter += 1
