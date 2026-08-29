# PitchVision

**Multi-object tracking on soccer footage — with the tracker written from scratch, not imported.**

Detection tells you *there is a player in this frame*. Tracking tells you *it's the same player as last frame*. That second problem is the hard one, and it's the one this project implements directly: a Kalman filter per track for motion prediction, IoU-based association, and the Hungarian algorithm for optimal assignment between predictions and new detections.

YOLOv11 supplies the detections. Everything downstream is hand-rolled.

---

## Why this exists

You can `pip install deep-sort-realtime` and have working tracking in five minutes. This project doesn't, because the point was to understand the machinery:

| Component | Implementation |
|---|---|
| Detection | YOLOv11 (`yolo11n`), filtered to person + sports ball |
| Motion model | `filterpy` Kalman filter — constant-velocity state per track, predicting the next bounding box |
| Association | IoU cost matrix between predicted and detected boxes |
| Assignment | `scipy.optimize.linear_sum_assignment` — Hungarian algorithm, globally optimal matching |
| Track lifecycle | `Track` class with hit counts, age, and `max_age` / `min_hits` gating for birth and death |

`class Track` and `class DeepSORT` in the notebook are the whole tracker.

The same three ideas — predict state, associate measurements, manage track identity — are what multi-object tracking looks like in robotics and autonomous driving. Soccer players are just a convenient set of fast-moving, mutually-occluding targets.

---

## The tracking problem, concretely

Twenty-two players in similar uniforms, constantly occluding each other, moving unpredictably. Every frame you must decide: is this detection a player you already know, or a new one? Get it wrong and identities swap.

The pipeline each frame:

```
frame ──→ YOLOv11 ──→ detections
                          │
  existing tracks ──→ Kalman predict ──→ predicted boxes
                          │
                    IoU cost matrix
                          │
               Hungarian assignment
                          │
        ┌─────────────────┼─────────────────┐
   matched            unmatched          unmatched
   → Kalman update    detections         tracks
                      → new track        → age out at max_age
```

Tuning `max_age` (how long to keep a track alive without detections) and `min_hits` (how many detections before a track is confirmed) is the difference between flickering IDs and phantom tracks that never die.

---

## Outputs

- `outputs/soccer_tracked.mp4` — rendered footage with persistent per-player bounding boxes and track IDs
- `outputs/Tracking Diagram.svg` — architecture diagram of the pipeline

The rendered video is 64 MB and not committed. Re-render it by running the notebook, or see the diagram for the pipeline structure.

---

## Quickstart

**Requirements:** Python 3.10–3.13, NVIDIA GPU recommended.

```bash
# PyTorch FIRST — pick the CUDA build for your GPU
# (Blackwell / RTX 50-series needs CUDA 12.8+)
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt
jupyter notebook pitchvision.ipynb
```

### Getting the data

Input footage is not committed (12 MB, third-party). The notebook expects a video at `data/`. Any soccer clip works — update `VIDEO_PATH` in the notebook to point at yours. `yolo11n.pt` downloads automatically on first run.

---

## Stack

Python 3.12 · Ultralytics YOLOv11 · filterpy (Kalman) · SciPy (Hungarian assignment) · OpenCV · NumPy · matplotlib

## Limitations

- **No appearance embeddings.** Real DeepSORT adds a re-identification CNN so that a track surviving a long occlusion can be matched by appearance, not just position. This implementation is motion-and-overlap only, which means identities can swap when players cross closely.
- Constant-velocity motion model — it does not anticipate direction changes, and soccer players change direction constantly
- Fixed camera assumption; no compensation for pan or zoom
- No homography to real pitch coordinates, so positions are in image space rather than metres
- Not evaluated against ground-truth annotations — no MOTA/IDF1 numbers, so quality is assessed visually

The appearance-embedding gap is the most interesting one to close next; it's what separates SORT from DeepSORT proper.

## Provenance

Built for CS 370 (Introduction to AI) at NJIT, Fall 2025, and cleaned up since. The tracker implementation is my own.
