# Object Tracker

Real-time single-object tracking in aerial video. You pick one pixel on the first
frame, and the tracker follows that object through the video -- including losing it
when it leaves the frame and re-acquiring it when it returns. Everything runs on the
CPU (no GPU needed) at 30+ fps on 1920x1080 video.

**Demo video:** [watch here](https://drive.google.com/file/d/1Lnu8d8e9A5EVREzwU8Fwh8y3mXXoDdCR)

**Project summary:** [read here](https://drive.google.com/file/d/16XAmiay_IxqWHRcJ0dj1QMTJJE1jKzoR)

## How to run -- one double-click

> You do not need to open or run any code file.

- **Windows:** double-click **`run_tracker.bat`**
- **macOS:** double-click **`run_tracker.command`**

That's it. On the first run it sets everything up by itself (including downloading
and installing Python if the computer doesn't have it -- one time, a few minutes,
no admin rights needed). After that it starts instantly.

## Using the app

1. **Choose a video** -- paste a URL or a local file path, or click *Browse...*.
   Only video files are accepted.
2. **Pick the pixel on the first frame** -- click it with the mouse, or type the
   X coordinate, ENTER, the Y coordinate, ENTER. The mark is shown on the frame;
   press ENTER again to start tracking.
3. **Watch the tracking** -- a green box follows the selected object. When the
   object leaves the frame the tracker searches for it and re-locks when it returns.

Keys:

| Key | Where | What it does |
|---|---|---|
| `S` | selection screen | toggle saving the run to an output video (`outputs/`) |
| `r` | selection screen | reset the current selection |
| `ESC` | selection screen | back to video selection |
| `q` / `ESC` | during tracking | stop the run |
| `R` | end screen | run again (the video address is kept) |
| `ESC` | end screen | exit |

## Requirements

| | Version used |
|---|---|
| Python | 3.12 (installed automatically if missing) |
| opencv-python | 5.0.0.93 |
| numpy | 2.5.0 |

The VitTrack tracking model (`models/object_tracking_vittrack_2023sep.onnx`) is
included in the repository. `tkinter` (part of the standard Python installation) is
used for the small file dialog.

Manual setup, if you prefer doing it yourself:

```
python -m venv venv
venv/Scripts/pip install -r requirements.txt      (Windows)
venv/bin/pip install -r requirements.txt          (macOS / Linux)
venv/Scripts/python src/app.py
```

## Command line

The tracker can also run directly, without the app:

```
python src/tracker.py --video "path/or/URL" --x 971 --y 533
```

Useful flags: `--save-video` records the tracking display to `outputs/`,
`--verbose` prints per-frame diagnostics, `--box-size N` changes the tracking
box size (default 40).

## How it works (short version)

The on-screen overlay (crosshair and frame lines) is removed by inpainting before any
processing, so the tracker never sees it. Tracking itself is built around OpenCV's
VitTrack (a lightweight CPU tracker) with a validity check on its confidence score and
box geometry. When the target is declared lost, the camera's global motion (optical
flow + RANSAC) keeps predicting where the object should be, and an ORB feature search
around that prediction re-detects it when it comes back -- matched at the right scale
using the camera zoom the system measures by itself, and verified against a small set
of "supporter" points tracked in the object's neighborhood, so the tracker re-locks on
the right object and not on a look-alike. A full description of the approach, the
alternatives that were tried, and the known limitations is in the project summary
document linked above.

## Project structure

| Path | Role |
|---|---|
| `run_tracker.bat` / `run_tracker.command` | **start here** -- one-click launchers (Windows / macOS) |
| `src/app.py` | the user interface (video selection, pixel picking, run loop) |
| `src/tracker.py` | the tracking pipeline itself |
| `src/overlay.py` | overlay removal (masking + inpainting) and screen-fit helpers |
| `src/loss_detection.py` | validity checks for the tracker's output |
| `src/gmc.py` | global camera-motion estimation |
| `models/` | the VitTrack ONNX model |
| `requirements.txt` | Python package versions |
