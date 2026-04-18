import numpy as np
import cv2


# ==========================
# VISUAL STYLE (BGR)
# ==========================
PLAYER_FALLBACK_COLOR = (0, 0, 255)    # Red (used if team_color missing)
REFEREE_COLOR         = (0, 0, 0)      # Black
BALL_COLOR            = (255, 0, 0)    # Blue
TEXT_COLOR            = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==========================
# GEOMETRY
# ==========================
def feet_anchor(bbox):
    """
    Compute ground contact point slightly below bbox bottom
    to simulate foot contact with pitch.
    """
    x1, y1, x2, y2 = map(int, bbox)
    return (x1 + x2) // 2, y2 + 2


def ellipse_axes_from_bbox(bbox, frame_height):
    """
    Perspective-aware broadcast footprint.
    Wider near camera, flatter far away.
    """
    x1, _, x2, y2 = map(int, bbox)
    bbox_width = max(12, x2 - x1)

    perspective = min(1.0, y2 / frame_height)

    axis_x = int(bbox_width * (0.60 + 0.25 * perspective))
    axis_y = int(axis_x * 0.22)  # very flat ground ellipse

    return axis_x, axis_y


# ==========================
# DRAWING PRIMITIVES
# ==========================
def draw_ground_ellipse(frame, center, axes, color):
    """
    Draw a soft, broadcast-style ground ellipse.
    Color MUST be BGR.
    """
    overlay = frame.copy()

    # Filled ellipse
    cv2.ellipse(
        overlay,
        center=center,
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=color,
        thickness=-1
    )

    # Alpha blend
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Thin white outline
    cv2.ellipse(
        frame,
        center=center,
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(255, 255, 255),
        thickness=1
    )


def draw_id_label(frame, text, center, bg_color=(0, 0, 0)):
    """
    Draw track ID label centered above the ellipse.
    Background color can be team color.
    """
    scale = 0.45
    thickness = 2

    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)

    bg_tl = (center[0] - w // 2 - 4, center[1] - h // 2 - 4)
    bg_br = (center[0] + w // 2 + 4, center[1] + h // 2 + 4)

    cv2.rectangle(frame, bg_tl, bg_br, bg_color, -1)

    cv2.putText(
        frame,
        text,
        (center[0] - w // 2, center[1] + h // 2),
        FONT,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA
    )


def draw_inverted_triangle(frame, center, size):
    """
    Inverted triangle marking ball ground contact.
    """
    cx, cy = center

    pts = np.array([
        [cx, cy + size],
        [cx - size, cy - size],
        [cx + size, cy - size]
    ], dtype=np.int32)

    cv2.fillPoly(frame, [pts], BALL_COLOR)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1)



def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)