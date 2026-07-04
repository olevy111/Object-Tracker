"""
Usage:
    python stage3_loss_detection.py --video "../ex/track-train.mp4"
    python stage3_loss_detection.py --video "../ex/track-train.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2
import numpy as np

from stage1_overlay import OverlayCleaner, pick_point_by_click

BOX_SIZE = 40
WINDOW_NAME = "Stage 3 - Loss Detection"
DEFAULT_MODEL = "models/object_tracking_vittrack_2023sep.onnx"

SCORE_THRESHOLD = 0.30
LOST_N_FRAMES = 3
RECOVER_N_FRAMES = 2
MAX_BBOX_AREA_FRAC = 0.03
MAX_BBOX_ASPECT_RATIO = 2.0

TEMPLATE_MATCH_THRESHOLD = 0.10  # kept low; NCC signal is weak
GOOD_TEMPLATE_SCORE = 0.5


def is_valid(bbox, score, frame_area):
    if score < SCORE_THRESHOLD:
        return False
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if (w * h) > MAX_BBOX_AREA_FRAC * frame_area:
        return False
    if max(w, h) > MAX_BBOX_ASPECT_RATIO * min(w, h):
        return False
    return True


def crop_template(frame, bbox, size):
    x, y, w, h = [int(v) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size))


def template_similarity(frame, bbox, templates, size):
    """Returns -1.0 if bbox can't be cropped (fully off-frame)."""
    candidate = crop_template(frame, bbox, size)
    if candidate is None:
        return -1.0
    best = -1.0
    for tmpl in templates:
        if tmpl is None:
            continue
        result = cv2.matchTemplate(candidate, tmpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(result[0, 0]))
    return best


def draw_tracking(frame, bbox, score, color=(0, 255, 0)):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"TRACKING score={score:.2f}", (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_lost(frame, last_good_bbox, score, color=(0, 0, 255), dim_color=(120, 120, 120)):
    if last_good_bbox is not None:
        x, y, w, h = [int(v) for v in last_good_bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), dim_color, 1)
    cv2.putText(frame, f"LOST  score(stale)={score:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Stage 3: score-gated loss detection + appearance-verified recovery")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--x", type=int, default=None, help="Manual pixel x on frame 1")
    parser.add_argument("--y", type=int, default=None, help="Manual pixel y on frame 1")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to VitTrack ONNX model")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--lost-n", type=int, default=LOST_N_FRAMES)
    parser.add_argument("--recover-n", type=int, default=RECOVER_N_FRAMES)
    parser.add_argument("--max-bbox-area-frac", type=float, default=MAX_BBOX_AREA_FRAC)
    parser.add_argument("--template-match-threshold", type=float, default=TEMPLATE_MATCH_THRESHOLD)
    parser.add_argument("--show-cleaned", action="store_true")
    parser.add_argument("--show-mask", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height
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

    template_size = args.box_size
    orig_template = crop_template(cleaned1, init_box, template_size)
    recent_template = orig_template

    cv2.namedWindow(WINDOW_NAME)

    show_cleaned = args.show_cleaned
    show_mask = args.show_mask

    frame_idx = 0
    fps_window_start = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    cleaned = cleaned1
    frame = frame1

    state = "TRACKING"
    bad_count = 0
    good_count = 0
    last_good_bbox = init_box
    score = 1.0
    bbox = init_box

    while True:
        base = cleaned if show_cleaned else frame
        display = base.copy()

        if show_mask:
            display[cleaner.full_mask > 0] = (0, 0, 255)

        if state == "TRACKING":
            draw_tracking(display, bbox, score)
        else:
            draw_lost(display, last_good_bbox, score)

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
        _, bbox = tracker.update(cleaned)
        score = tracker.getTrackingScore()
        valid = is_valid(bbox, score, frame_area)

        if state == "TRACKING":
            if valid:
                bad_count = 0
                last_good_bbox = bbox
                if score >= GOOD_TEMPLATE_SCORE:
                    refreshed = crop_template(cleaned, bbox, template_size)
                    if refreshed is not None:
                        recent_template = refreshed
            else:
                bad_count += 1
                if bad_count >= args.lost_n:
                    state = "LOST"
                    good_count = 0
                    print(f"Frame {frame_idx} | TRACKING->LOST | score={score:.3f} | bbox={bbox}")
        else:  # LOST
            if valid:
                sim = template_similarity(cleaned, bbox, [orig_template, recent_template], template_size)
                if sim >= args.template_match_threshold:
                    good_count += 1
                    if good_count >= args.recover_n:
                        state = "TRACKING"
                        bad_count = 0
                        last_good_bbox = bbox
                        refreshed = crop_template(cleaned, bbox, template_size)
                        if refreshed is not None:
                            recent_template = refreshed
                        print(f"Frame {frame_idx} | LOST->TRACKING | score={score:.3f} | sim={sim:.3f} | bbox={bbox}")
                else:
                    good_count = 0
                    print(f"Frame {frame_idx} | LOST (appearance reject) | score={score:.3f} | sim={sim:.3f} | bbox={bbox}")
            else:
                good_count = 0

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
