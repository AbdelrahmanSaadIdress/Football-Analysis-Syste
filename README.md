# ⚽ Football Analysis System

**End-to-end broadcast-quality football video analysis powered by YOLOv8, ByteTrack, and computer vision.**

Detects players, referees, and the ball · assigns teams by jersey colour · tracks ball possession · estimates real-world speed and distance · compensates for camera pan and tilt.

---

<!-- ════════════════  INPUT VIDEO  ════════════════ -->
### Input
https://github.com/user-attachments/assets/06dbe34d-b1f8-452f-8d28-d2fb40832787
<!-- ════════════════  OUTPUT VIDEO  ════════════════ -->
### Output
https://github.com/user-attachments/assets/e76b15b6-0e19-4f17-b515-339963183776
</div>

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
4. [Installation](#installation)
5. [Model Weights](#model-weights)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Configuration](#configuration)
9. [Module Reference](#module-reference)

---

## Features

| Capability | Details |
|---|---|
| **Object Detection** | YOLOv8 fine-tuned on football footage — detects players, goalkeepers (merged to player class), referees, and ball |
| **Multi-Object Tracking** | ByteTrack assigns stable IDs across frames; goalkeepers re-labelled as players during tracking |
| **Ball Interpolation** | Linear interpolation of all four bbox coordinates fills gaps between detections; frames outside the detection window are intentionally left empty |
| **Team Assignment** | K-Means clustering on upper-body jersey colour; refreshed every N frames; goalkeeper overrides supported |
| **Ball Possession** | Nearest-player assignment per frame; cumulative possession percentage displayed as a broadcast-style HUD bar |
| **Camera Compensation** | Lucas–Kanade sparse optical flow on a top-ROI mask; estimated pan/tilt subtracted from every object position |
| **Perspective Transformation** | Homography from four manually-defined pitch corners maps image positions to real-world metres |
| **Speed & Distance** | World-space positions differentiated over a sliding 5-frame window; results shown as km/h and metres per player |
| **Broadcast Visualisation** | Layered ground ellipses, pill-shaped ID badges, glowing ball rings, possession-arrow indicators, dashed team-has-ball halos, and a bottom-centre possession HUD |

---

## System Architecture

```
Input Video
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        main.py  (pipeline orchestrator)         │
│                                                                 │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐    │
│  │   Tracker    │   │ TeamAssigner  │   │  CameraMovement  │    │
│  │  (YOLOv8 +   │   │ (KMeans on    │   │  (Lucas–Kanade   │    │
│  │  ByteTrack)  │   │  jersey RGB)  │   │   optical flow)  │    │
│  └──────┬───────┘   └───────┬───────┘   └────────┬─────────┘    │
│         │                   │                    │              │
│         ▼                   ▼                    ▼              │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐    │
│  │    Ball      │   │    Ball       │   │  Perspective     │    │
│  │ Interpolation│   │   Assigner    │   │  Transformer     │    │
│  └──────┬───────┘   └───────┬───────┘   └────────┬─────────┘    │
│         │                   │                    │              │
│         └───────────────────┴────────────────────┘              │
│                             │                                   │
│                             ▼                                   │
│                  ┌──────────────────┐                           │
│                  │ SpeedAndDistance │                           │
│                  └────────┬─────────┘                           │
│                           │                                     │
│                           ▼                                     │
│               ┌───────────────────────┐                         │
│               │  Visualisation Layer  │                         │
│               │  (draw_utils.py)      │                         │
│               └───────────────────────┘                         │ 
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
Output Video
```

---

## Full Pipeline Walkthrough

### Step 1 — Read Video

`utils/video_utils.py · read_video()`

OpenCV reads every frame into a list of NumPy arrays (`BGR`, `H × W × 3`). All downstream modules receive this list; the original frames are never mutated until the final visualisation pass.

---

### Step 2 — Detection & Tracking

`trackers/tracker.py · Tracker.get_object_tracks()`

**Detection** — YOLOv8 runs inference in batches of 20 frames. Goalkeepers are re-mapped to the `player` class before tracking so they receive stable player IDs.

**Tracking** — ByteTrack matches detections across frames using a Kalman filter + IoU-based assignment, producing a persistent integer `track_id` for each object.

**Output structure:**
```python
tracks = {
    "players":  [ {track_id: {"bbox": [x1,y1,x2,y2], "track_id": int}, ...}, ... ],  # one dict per frame
    "referees": [ {track_id: {"bbox": ...}, ...}, ... ],
    "balls":    [ {track_id: {"bbox": ...}, ...}, ... ],
}
```

Results are pickled to `runs/backups/tracks.pkl` so re-runs skip inference entirely (`read_backup=True`).

---

### Step 3 — Ball Position Interpolation

`trackers/tracker.py · Tracker.interpolate_ball_positions()`

The ball detector occasionally misses frames (occlusion, motion blur). Linear interpolation fills every gap **between** real detections:

- All four bbox coordinates (`x1, y1, x2, y2`) are interpolated independently so ball size is preserved.
- Frames before the first detection and after the last detection are **left empty** — the ball is not frozen at a stale position.
- Interpolated frames carry an `"interpolated": True` flag for downstream awareness.

```
Frame:    0    1    2   [3]   4    5    6   [7]   8    9
Detected: —    —    —    ✓    —    —    —    ✓    —    —
Filled:   —    —    —    ✓   fill fill fill   ✓    —    —
```

---

### Step 4 — Camera Movement Estimation & Compensation

`camera_movement/camera_movement.py · CameraMovement`

Broadcast cameras pan and tilt continuously. Raw pixel positions therefore conflate player movement with camera movement.

**Estimation:**
1. Shi-Tomasi corner features are detected in a top-strip ROI (the scoreboard area, which is static relative to the camera).
2. Lucas–Kanade optical flow tracks these features frame-to-frame.
3. The feature with the largest displacement is taken as `(cx, cy)` — the camera's translation vector for that frame.
4. If displacement exceeds a threshold the feature set is re-initialised from the new frame.

**Compensation** (`apply_camera_compensation`):
Every object's centre position is adjusted:
```python
adjusted_position = (cx_pixel - camera_dx, cy_pixel - camera_dy)
```
This `adjusted_position` is used by all metric calculations downstream.

---

### Step 5 — Team Assignment

`team_assigners/team_assigner.py · TeamAssigner`

1. **Colour extraction** — the upper half of each player's bounding box is cropped and 2-cluster K-Means separates jersey (foreground, brighter) from pitch/background pixels.
2. **Team clustering** — on the first frame all extracted colours are clustered into two groups; cluster centres become `team_colors[1]` and `team_colors[2]` (BGR tuples).
3. **Per-player prediction** — subsequent frames predict each new player's team from their jersey colour using the stored K-Means model. Results are cached by `track_id`.
4. **Refresh** — the cache is invalidated every `refresh_interval` frames to handle kit-change or lighting drift.
5. **Goalkeeper override** — specific `track_id` values can be hard-coded to a team to correct misclassifications.

---

### Step 6 — Ball Possession Assignment

`ball_assigners/ball_assigner.py · BallAssigner`
`trackers/tracker.py · Tracker.update_ball_possession()`

For each frame:
1. The ball's centre is computed from its bbox.
2. Euclidean distance is measured from the ball centre to every player's foot anchor.
3. The closest player within `max_distance` pixels is declared the ball carrier (`has_ball = True`).
4. All teammates of the ball carrier receive `team_has_ball = True`.
5. Possession frames are accumulated in `team_possession_frames` for the HUD percentage.

---

### Step 7 — Perspective Transformation

`perspective_transformer/perspective_transformer.py · PerspectiveTransformer`

Four manually-defined pixel coordinates mark the four corners of the visible pitch area. `cv2.getPerspectiveTransform` computes a homography **H** that maps image space → world space (metres):

```
Image corners (pixels)  →  World corners (metres)
[110, 1035]  →  [0,  68]   bottom-left
[265,  275]  →  [0,   0]   top-left
[910,  260]  →  [23.32, 0] top-right
[1640, 915]  →  [23.32,68] bottom-right
```

Each player's `adjusted_position` is transformed; points outside the pitch polygon return `None`.

---

### Step 8 — Speed & Distance

`speed_and_distance/speed_and_distance.py · SpeedAndDistance`

Over a sliding window of 5 frames:
```
distance (m) = Euclidean( world_pos[t], world_pos[t+5] )
time (s)     = 5 frames / 24 fps
speed (km/h) = (distance / time) × 3.6
```
Cumulative distance is tracked per player. Both values are written back into the `tracks` dict and drawn as text overlays below each player's bounding box.

---

### Step 9 — Visualisation

`utils/draw_utils.py` · `trackers/tracker.py · Tracker.visualize_tracks()`

Three rendering passes produce the final annotated frames:

| Pass | Function | What is drawn |
|---|---|---|
| `visualize_tracks` | draw_utils | Ground ellipses, ID badges, ball rings, possession arrows, team halos, possession HUD bar |
| `draw_camera_movement` | CameraMovement | Camera dx/dy readout overlay |
| `draw_speed_and_distance` | SpeedAndDistance | Speed (km/h) and distance (m) text per player |

**Visual elements:**

- **Player shadow disc** — perspective-aware ground ellipse filled with team colour; glow bloom layers for depth.
- **ID badge** — pill-shaped label centred on the ellipse, tinted to team colour.
- **Ball** — concentric glowing red rings with translucent fill and specular shine arc.
- **Ball-carrier marker** — red downward-pointing chevron above the player's head with drop shadow.
- **Team possession halo** — dashed red ring around foot ellipses of all teammates of the ball carrier.
- **Possession HUD** — bottom-centre pill bar split by team colour with live percentage labels.

---

### Step 10 — Save Output

`utils/video_utils.py · save_video()`

OpenCV `VideoWriter` (codec `mp4v`, 30 fps) writes the annotated frame list to the output path. The output directory is created automatically.

---

## Installation

```bash
git clone https://github.com/AbdelrahmanSaadIdress/Football-Analysis-System_.git
cd Football-Analysis-System_

pip install ultralytics supervision scikit-learn opencv-python numpy
```

> Python 3.8 or newer is required. A CUDA-capable GPU is strongly recommended for real-time inference speeds.

---

## Model Weights

The fine-tuned YOLOv8 model is hosted on Kaggle:

> **[⬇ Download `best.pt` from Kaggle](https://www.kaggle.com/models/abdelrhmansaadidrees/football-match-analysis-system)**

After downloading, place the file at:

```
Football-Analysis-System_/
└── yolo_models/
    └── best.pt          ← put it here
```

Or pass a custom path at runtime with `--model_path`.

---

## Quick Start

```bash
# 1. Download model weights (see above) and place in yolo_models/best.pt

# 2. Put your input video at the default path
cp your_match.mp4 video_samples/08fd33_4.mp4

# 3. Run the pipeline
python main.py

# Output is written to:
#   runs/outputs/result.mp4
```

**Custom paths:**

```bash
python main.py \
  --model_path  yolo_models/best.pt \
  --output_path runs/outputs/my_match.mp4
```

> On the first run, tracks are computed and cached to `runs/backups/tracks.pkl`.  
> Subsequent runs load the cache instantly — delete the file to force re-detection.

---

## Project Structure

```
Football-Analysis-System_/
│
├── main.py                          # Pipeline orchestrator & config
│
├── trackers/
│   └── tracker.py                   # YOLOv8 detection, ByteTrack, interpolation, visualisation
│
├── team_assigners/
│   └── team_assigner.py             # K-Means jersey colour clustering
│
├── ball_assigners/
│   └── ball_assigner.py             # Nearest-player possession assignment
│
├── camera_movement/
│   └── camera_movement.py           # Lucas–Kanade optical flow & compensation
│
├── perspective_transformer/
│   └── perspective_transformer.py   # Homography → world coordinates (metres)
│
├── speed_and_distance/
│   └── speed_and_distance.py        # Speed (km/h) and cumulative distance (m)
│
├── utils/
│   ├── video_utils.py               # read_video / save_video
│   └── draw_utils.py                # All drawing primitives (ellipses, badges, HUD)
│
├── video_samples/
│   └── 08fd33_4.mp4                 # Default input clip
│
├── yolo_models/
│   └── best.pt                      # Model weights (download separately)
│
└── runs/
    ├── backups/tracks.pkl           # Cached detections (auto-created)
    └── outputs/result.mp4           # Output video (auto-created)
```

---

## Configuration

All tunable constants live at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `video_samples/08fd33_4.mp4` | Input video path |
| `SAVE_PATH` | `runs/outputs/result.mp4` | Output video path |
| `COURT_WIDTH` | `68` | Pitch width in metres |
| `COURT_LENGTH` | `23.32` | Visible pitch length in metres |
| `PIXEL_VERTICES` | see file | Four pitch-corner pixel coordinates — **must be re-calibrated for each camera angle** |

In `team_assigner.py`:

| Constant | Default | Description |
|---|---|---|
| `refresh_interval` | `30` | Frames between jersey-colour cache refreshes |
| `goalkeeper_ids` | `{99: 2, ...}` | Hard-coded track-ID → team overrides for goalkeepers |

In `tracker.py`:

| Constant | Default | Description |
|---|---|---|
| `batch_size` | `20` | Frames per YOLO inference batch |
| `ball_max_distance` | `300` | Max pixel distance for ball-to-player assignment |

---

## Module Reference

| Module | Class / Function | Responsibility |
|---|---|---|
| `trackers` | `Tracker` | Detection, tracking, interpolation, possession, visualisation |
| `team_assigners` | `TeamAssigner` | K-Means jersey clustering + per-player team prediction |
| `ball_assigners` | `BallAssigner` | Frame-level ball → nearest player assignment |
| `camera_movement` | `CameraMovement` | Optical flow estimation; `apply_camera_compensation` free function |
| `perspective_transformer` | `PerspectiveTransformer` | Homography; pixel → world-space point transform |
| `speed_and_distance` | `SpeedAndDistance` | Sliding-window speed & cumulative distance |
| `utils` | `draw_utils` | All CV2 drawing primitives and broadcast HUD |
| `utils` | `video_utils` | `read_video` / `save_video` |
