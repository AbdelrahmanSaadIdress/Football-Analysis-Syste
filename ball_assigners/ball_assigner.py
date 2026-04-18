# ball_assigners.py
import numpy as np
from utils import feet_anchor


class BallAssigner:
    def __init__(self, max_distance=300):
        self.max_distance = max_distance

    def assign_ball_to_player(self, players_tracks, ball_tracks):
        """
        Returns:
            assigned_player_id (int) or None
        """

        if not ball_tracks or not players_tracks:
            return None

        # Use first detected ball
        _, ball_data = next(iter(ball_tracks.items()))
        ball_center = feet_anchor(ball_data["bbox"])

        min_dist = float("inf")
        assigned_player = None

        for track_id, data in players_tracks.items():
            player_center = feet_anchor(data["bbox"])
            dist = np.linalg.norm(
                np.array(ball_center) - np.array(player_center)
            )

            if dist < min_dist and dist < self.max_distance:
                min_dist = dist
                assigned_player = track_id

            

        return assigned_player
