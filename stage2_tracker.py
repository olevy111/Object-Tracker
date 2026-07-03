"""
Stage 2 -- Short-term core tracker.

Builds on Stage 1. Initializes an OpenCV TrackerVit (CPU-capable, exposes a
per-frame confidence score) on the CLEANED frame at the clicked pixel --
NEVER the original frame, so the template is crosshair-free. Every update
also runs on the CLEANED frame. Draws the tracked box and prints the
confidence score every frame.

No loss handling yet (that's Stage 3) -- the box is drawn every frame
regardless of the tracker's `ok` flag or score, just colored to show status:
    green  = tracker reports ok
    orange = tracker reports NOT ok (still drawn, not acted on yet)

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    q / ESC - quit

Usage:
    python stage2_tracker.py --video "../ex/track-train.mp4"
    python stage2_tracker.py --video "../ex/track-train.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2

from stage1_overlay import OverlayCleaner, pick_point_by_click

BOX_SIZE = 40
WINDOW_NAME = "Stage 2 - Core Tracker"
DEFAULT_MODEL = "models/object_tracking_vittrack_2023sep.onnx"


def draw_tracked_box(frame, bbox, ok, score, color_ok=(0, 255, 0), color_bad=(0, 165, 255)):
    x, y, w, h = [int(v) for v in bbox]
    color = color_ok if ok else color_bad
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"score {score:.2f}", (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Stage 2: core tracker on cleaned frames")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--x", type=int, default=None, help="Manual pixel x on frame 1")
    parser.add_argument("--y", type=int, default=None, help="Manual pixel y on frame 1")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE,
                         help="Side length in px of the tracker init box")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to VitTrack ONNX model")
    parser.add_argument("--show-cleaned", action="store_true",
                         help="Start with the cleaned stream shown instead of original")
    parser.add_argument("--show-mask", action="store_true",
                         help="Start with the mask-visualization overlay on")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video}")
    print(f"Resolution: {width}x{height}, source FPS: {src_fps:.2f}")

    cleaner = OverlayCleaner(width, height)

    ok, frame1 = cap.read()
    if not ok:
        print("ERROR: could not read first frame.")
        sys.exit(1)

    if args.x is not None and args.y is not None:
        point = (args.x, args.y)
        print(f"Using manual point: {point}")
    else:
        point = pick_point_by_click(frame1)
        print(f"Picked point: {point}")

    cleaned1 = cleaner.clean(frame1)

    half = args.box_size // 2
    init_box = (point[0] - half, point[1] - half, args.box_size, args.box_size)

    params = cv2.TrackerVit_Params()
    params.net = args.model
    tracker = cv2.TrackerVit_create(params)
    tracker.init(cleaned1, init_box)
    print(f"Tracker initialized on cleaned frame, box={init_box}")

    cv2.namedWindow(WINDOW_NAME)

    show_cleaned = args.show_cleaned
    show_mask = args.show_mask

    frame_idx = 0
    fps_window_start = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    cleaned = cleaned1
    frame = frame1
    tracker_ok, bbox, score = True, init_box, 1.0

    while True:
        base = cleaned if show_cleaned else frame
        display = base.copy()

        if show_mask:
            display[cleaner.full_mask > 0] = (0, 0, 255)

        draw_tracked_box(display, bbox, tracker_ok, score)

        stream_label = "CLEANED" if show_cleaned else "ORIGINAL"
        mask_label = " + MASK" if show_mask else ""
        cv2.putText(display, f"frame {frame_idx}  fps {display_fps:.1f}  [{stream_label}{mask_label}]",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            show_cleaned = not show_cleaned
        elif key == ord("m"):
            show_mask = not show_mask

        ok, frame = cap.read()
        if not ok:
            print("End of video.")
            break
        frame_idx += 1

        cleaned = cleaner.clean(frame)
        tracker_ok, bbox = tracker.update(cleaned)
        score = tracker.getTrackingScore()
        print(f"Frame {frame_idx} | ok={tracker_ok} | score={score:.3f} | bbox={bbox}")

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            display_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_window_start = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
