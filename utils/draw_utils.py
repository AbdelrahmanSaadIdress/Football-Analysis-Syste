import numpy as np
import cv2


# ============================================================
# BROADCAST DESIGN TOKENS  (all BGR)
# ============================================================

PLAYER_FALLBACK_COLOR = (60, 60, 220)
REFEREE_COLOR         = (30, 30, 30)
TEXT_COLOR            = (255, 255, 255)
FONT                  = cv2.FONT_HERSHEY_DUPLEX

BALL_RING_OUTER  = (40,  40, 255)
BALL_RING_INNER  = (80,  80, 255)
BALL_FILL        = (10,  10, 180)
BALL_SHINE       = (200, 220, 255)

POSS_RING_COLOR  = (0,  20, 230)
POSS_GLOW_COLOR  = (0,   0, 130)
POSS_DASH_COLOR  = (40,  60, 255)

ARROW_FILL       = (0,  10, 240)
ARROW_BORDER     = (60, 70, 255)
ARROW_SHINE      = (180, 200, 255)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def feet_anchor(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return (x1 + x2) // 2, y2 + 2


def ellipse_axes_from_bbox(bbox, frame_height):
    x1, _, x2, y2 = map(int, bbox)
    bbox_width = max(12, x2 - x1)
    perspective = min(1.0, y2 / frame_height)
    axis_x = int(bbox_width * (0.58 + 0.24 * perspective))
    axis_y = max(4, int(axis_x * 0.24))
    return axis_x, axis_y


def _alpha_blend(frame, overlay, alpha):
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# ============================================================
# PLAYER GROUND ELLIPSE
# ============================================================

def draw_ground_ellipse(frame, center, axes, color):
    ax, ay = axes

    # Layer 1: diffuse outer glow
    ov = frame.copy()
    for expand in (8, 5, 3):
        cv2.ellipse(ov, center, (ax + expand, ay + expand),
                    0, 0, 360, color, 2)
    _alpha_blend(frame, ov, 0.18)

    # Layer 2: filled translucent disc
    ov2 = frame.copy()
    cv2.ellipse(ov2, center, (ax, ay), 0, 0, 360, color, -1)
    _alpha_blend(frame, ov2, 0.55)

    # Layer 3: crisp double rim
    cv2.ellipse(frame, center, (ax, ay), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.ellipse(frame, center, (ax, ay), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# PLAYER ID BADGE  (pill-shaped)
# ============================================================

def draw_id_label(frame, text, center, bg_color=(20, 20, 20)):
    scale     = 0.42
    thickness = 1
    pad_x, pad_y = 6, 4

    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)

    cx, cy   = center
    pill_w   = tw + pad_x * 2
    pill_h   = th + pad_y * 2 + baseline
    radius   = pill_h // 2

    tl = (cx - pill_w // 2, cy - pill_h // 2)
    br = (cx + pill_w // 2, cy + pill_h // 2)

    ov = frame.copy()
    cv2.rectangle(ov, (tl[0] + radius, tl[1]), (br[0] - radius, br[1]), bg_color, -1)
    cv2.circle(ov, (tl[0] + radius, cy), radius, bg_color, -1)
    cv2.circle(ov, (br[0] - radius, cy), radius, bg_color, -1)
    _alpha_blend(frame, ov, 0.82)

    cv2.rectangle(frame, (tl[0] + radius, tl[1]), (br[0] - radius, br[1]), (255, 255, 255), 1)
    cv2.circle(frame, (tl[0] + radius, cy), radius, (255, 255, 255), 1)
    cv2.circle(frame, (br[0] - radius, cy), radius, (255, 255, 255), 1)

    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


# ============================================================
# BALL — glowing red rings
# ============================================================

def draw_ball_ellipse(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = max(14, (x2 - x1) // 2 + 10)
    ry = max(12, (y2 - y1) // 2 + 10)

    # Outer bloom glow
    ov = frame.copy()
    for r_add in (16, 11, 6):
        cv2.ellipse(ov, (cx, cy), (rx + r_add, ry + r_add),
                    0, 0, 360, BALL_RING_OUTER, 2)
    _alpha_blend(frame, ov, 0.30)

    # Translucent fill
    ov2 = frame.copy()
    cv2.ellipse(ov2, (cx, cy), (rx, ry), 0, 0, 360, BALL_FILL, -1)
    _alpha_blend(frame, ov2, 0.22)

    # Double solid ring
    cv2.ellipse(frame, (cx, cy), (rx + 2, ry + 2), 0, 0, 360, BALL_RING_OUTER, 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (rx,     ry    ), 0, 0, 360, BALL_RING_INNER, 2, cv2.LINE_AA)

    # Specular shine arc
    cv2.ellipse(frame, (cx, cy), (rx - 5, ry - 4), -30, 200, 320, BALL_SHINE, 1, cv2.LINE_AA)

    # Centre dot
    cv2.circle(frame, (cx, cy), 2, BALL_RING_OUTER, -1, cv2.LINE_AA)


# ============================================================
# PLAYER-HAS-BALL — downward chevron above head
# ============================================================

def draw_has_ball_triangle(frame, bbox):
    x1, y1, x2, _ = map(int, bbox)
    cx   = (x1 + x2) // 2
    size = max(14, (x2 - x1) // 5)

    gap    = 8
    tip_y  = y1 - gap
    base_y = tip_y - size * 2
    half_w = int(size * 1.1)

    pts = np.array([
        [cx,           tip_y ],
        [cx - half_w,  base_y],
        [cx + half_w,  base_y],
    ], dtype=np.int32)

    # Drop shadow
    shadow = pts + np.array([[2, 3]], dtype=np.int32)
    ov = frame.copy()
    cv2.fillPoly(ov, [shadow], (0, 0, 40))
    _alpha_blend(frame, ov, 0.50)

    # Filled body
    cv2.fillPoly(frame, [pts], ARROW_FILL)

    # Outer border
    cv2.polylines(frame, [pts], True, ARROW_BORDER, 2, cv2.LINE_AA)

    # Shine line
    mid_left = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2)
    cv2.line(frame, (pts[1][0] + 3, pts[1][1] + 3), mid_left, ARROW_SHINE, 1, cv2.LINE_AA)

    # Tip dot
    cv2.circle(frame, (cx, tip_y), 2, (255, 255, 255), -1, cv2.LINE_AA)


# ============================================================
# TEAM-HAS-BALL — dashed red possession halo
# ============================================================

def draw_team_has_ball_ellipse(frame, center, axes):
    ax, ay = axes[0] + 9, axes[1] + 7

    # Outer bloom
    ov = frame.copy()
    cv2.ellipse(ov, center, (ax + 8, ay + 6), 0, 0, 360, POSS_GLOW_COLOR, 4)
    _alpha_blend(frame, ov, 0.28)

    # Medium ring
    ov2 = frame.copy()
    cv2.ellipse(ov2, center, (ax + 4, ay + 3), 0, 0, 360, POSS_RING_COLOR, 2)
    _alpha_blend(frame, ov2, 0.55)

    # Crisp inner ring
    cv2.ellipse(frame, center, (ax, ay), 0, 0, 360, POSS_RING_COLOR, 2, cv2.LINE_AA)

    # Dashed accent arcs
    for start_angle in (20, 110, 200, 290):
        cv2.ellipse(frame, center, (ax, ay), 0,
                    start_angle, start_angle + 35,
                    POSS_DASH_COLOR, 2, cv2.LINE_AA)

    # Shimmer
    cv2.ellipse(frame, center, (ax - 3, ay - 2), 0, 185, 355,
                (200, 210, 255), 1, cv2.LINE_AA)


# ============================================================
# BROADCAST POSSESSION HUD
# ============================================================

def draw_possession_hud(frame, team_possession_frames, team_colors=None):
    fh, fw = frame.shape[:2]

    bar_w = min(420, fw - 60)
    bar_h = 52
    bar_x = (fw - bar_w) // 2
    bar_y = fh - bar_h - 18

    total = sum(team_possession_frames.values()) + 1e-9
    teams = sorted(team_possession_frames.keys())
    if not teams:
        return

    DEFAULT_COLORS = {}
    if len(teams) > 0:
        DEFAULT_COLORS[teams[0]] = (220, 80, 30)
    if len(teams) > 1:
        DEFAULT_COLORS[teams[1]] = (30, 180, 220)

    # Dark pill background
    radius = bar_h // 2
    ov = frame.copy()
    cv2.rectangle(ov, (bar_x + radius, bar_y),
                  (bar_x + bar_w - radius, bar_y + bar_h), (18, 18, 18), -1)
    cv2.circle(ov, (bar_x + radius,         bar_y + bar_h // 2), radius, (18, 18, 18), -1)
    cv2.circle(ov, (bar_x + bar_w - radius, bar_y + bar_h // 2), radius, (18, 18, 18), -1)
    _alpha_blend(frame, ov, 0.78)

    # Pill border
    cv2.rectangle(frame, (bar_x + radius, bar_y),
                  (bar_x + bar_w - radius, bar_y + bar_h), (80, 80, 80), 1)
    cv2.circle(frame, (bar_x + radius,         bar_y + bar_h // 2), radius, (80, 80, 80), 1)
    cv2.circle(frame, (bar_x + bar_w - radius, bar_y + bar_h // 2), radius, (80, 80, 80), 1)

    # Possession fill bars
    inner_x = bar_x + 10
    inner_w = bar_w - 20
    inner_y = bar_y + 14
    inner_h = bar_h - 28

    x_cursor = inner_x
    pcts = []
    for team in teams:
        pct = team_possession_frames[team] / total
        pcts.append(pct)
        seg_w = int(inner_w * pct)
        col   = (team_colors or {}).get(team, DEFAULT_COLORS.get(team, (150, 150, 150)))
        ov3 = frame.copy()
        cv2.rectangle(ov3, (x_cursor, inner_y),
                      (x_cursor + seg_w, inner_y + inner_h), col, -1)
        _alpha_blend(frame, ov3, 0.85)
        x_cursor += seg_w

    # Split divider
    split_x = inner_x + int(inner_w * pcts[0]) if pcts else inner_x + inner_w // 2
    cv2.line(frame, (split_x, bar_y + 6), (split_x, bar_y + bar_h - 6),
             (255, 255, 255), 1, cv2.LINE_AA)

    # Percentage labels
    label_y = bar_y + bar_h // 2 + 5
    for i, team in enumerate(teams):
        pct_str = f"{pcts[i] * 100:.0f}%"
        (tw, _), _ = cv2.getTextSize(pct_str, FONT, 0.48, 1)
        col = (team_colors or {}).get(team, DEFAULT_COLORS.get(team, (200, 200, 200)))
        lx = bar_x + 16 if i == 0 else bar_x + bar_w - tw - 16
        cv2.putText(frame, pct_str, (lx + 1, label_y + 1), FONT, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, pct_str, (lx,     label_y    ), FONT, 0.48, col,        1, cv2.LINE_AA)

    # "POSSESSION" title
    title = "POSSESSION"
    (ttw, _), _ = cv2.getTextSize(title, FONT, 0.36, 1)
    tx = (fw - ttw) // 2
    ty = bar_y - 6
    cv2.putText(frame, title, (tx + 1, ty + 1), FONT, 0.36, (0, 0, 0),       2, cv2.LINE_AA)
    cv2.putText(frame, title, (tx,     ty    ), FONT, 0.36, (180, 180, 180), 1, cv2.LINE_AA)


# ============================================================
# LEGACY COMPAT
# ============================================================

def draw_inverted_triangle(frame, center, size):
    cx, cy = center
    pts = np.array([
        [cx,         cy + size],
        [cx - size,  cy - size],
        [cx + size,  cy - size],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 200))
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1)


# ============================================================
# UTILITY
# ============================================================

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)