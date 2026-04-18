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


def draw_ball_ellipse(frame, bbox):
    """
    Draw a beautiful glowing red ellipse around the ball.
    Uses the bounding box to size the ellipse naturally around the ball.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Size ellipse to bbox with some padding
    rx = max(12, (x2 - x1) // 2 + 8)
    ry = max(10, (y2 - y1) // 2 + 8)

    RED = (0, 0, 220)          # BGR red
    RED_BRIGHT = (30, 30, 255) # brighter red highlight
    WHITE = (255, 255, 255)

    # Layer 1 – outer soft glow (semi-transparent, wide)
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy), (rx + 8, ry + 8), 0, 0, 360, (0, 0,160), 3)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Layer 2 – filled inner ellipse (semi-transparent)
    overlay2 = frame.copy()
    cv2.ellipse(overlay2, (cx, cy), (rx, ry), 0, 0, 360, RED, -1)
    cv2.addWeighted(overlay2, 0.25, frame, 0.75, 0, frame)

    # Layer 3 – solid red border
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, RED_BRIGHT, 2, cv2.LINE_AA)

    # Layer 4 – white inner highlight arc (top portion for shine)
    cv2.ellipse(frame, (cx, cy), (rx - 4, ry - 3), -20, 200, 340, WHITE, 1, cv2.LINE_AA)

def draw_has_ball_triangle(frame, bbox):
    """
    Draw a beautiful red upward-pointing triangle above the player's head.
    bbox is the player bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2

    # Position triangle above the head with a small gap
    tip_y   = y1 - 18         # apex (top point)
    base_y  = y1 - 4          # base of triangle (just above head)
    half_w  = 13               # half-width of base

    pts = np.array([
        [cx,          tip_y],   # apex
        [cx - half_w, base_y],  # bottom-left
        [cx + half_w, base_y],  # bottom-right
    ], dtype=np.int32)

    RED        = (0, 0, 210)
    RED_BRIGHT = (30, 30, 255)
    WHITE      = (255, 255, 255)

    # Drop shadow
    shadow_pts = pts + np.array([[2, 2]], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [shadow_pts], (0, 0, 60))
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Filled red triangle
    cv2.fillPoly(frame, [pts], RED)

    # Bright red outline
    cv2.polylines(frame, [pts], True, RED_BRIGHT, 2, cv2.LINE_AA)

    # White inner highlight line (left edge)
    cv2.line(frame,
            (pts[0][0], pts[0][1]),
            (pts[1][0], pts[1][1]),
            WHITE, 1, cv2.LINE_AA)


def draw_team_has_ball_ellipse(frame, center, axes):
    """
    Draw a beautiful glowing red ellipse around the foot of the player
    whose team is in possession.
    """
    RED       = (0, 0, 200)
    RED_GLOW  = (0, 0, 140)
    WHITE     = (255, 255, 255)

    ax, ay = axes[0] + 7, axes[1] + 5

    # Outer glow ring
    overlay = frame.copy()
    cv2.ellipse(overlay, center, (ax + 5, ay + 4), 0, 0, 360, RED_GLOW, 3)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Main red ellipse border
    cv2.ellipse(frame, center, (ax, ay), 0, 0, 360, RED, 2, cv2.LINE_AA)

    # White accent arc on top for shimmer
    cv2.ellipse(frame, center, (ax - 3, ay - 2), 0, 190, 350, WHITE, 1, cv2.LINE_AA)


def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)