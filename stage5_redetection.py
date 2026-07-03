"""
Stage 5 -- Re-detection.

Builds on Stage 3 (score-gated loss detection + hysteresis + appearance
templates). Replaces the passive "keep calling tracker.update() and hope it
stumbles back onto the object" behavior of Stages 3/4 with an ACTIVE search
while LOST.

Why this stage exists (from real testing, not guesswork): Stage 3 testing
showed that once genuinely LOST, the tracker's own continued updates only
recover the object if it's still within the tracker's local search window --
if the object actually leaves the frame and re-enters elsewhere, or drifts
far enough, passive updates never find it again, no matter how long you
wait. Stage 4 (GMC) was measured to not fix this either (VitTrack's search
range already covers this footage's camera motion; the losses aren't
caused by the object jumping outside a search window).

What's different here: while LOST, every frame, the cleaned frame's real
video-content region is actively searched for the saved reference
templates (the original click template, plus the most recent
high-confidence template) using multi-scale normalized cross-correlation
(cv2.matchTemplate at a handful of scales -- cheap because OpenCV's
matchTemplate is not a naive pixel-by-pixel sliding window, and a handful
of scales is far cheaper than a brute-force multi-scale sliding window
over every position). A candidate is only accepted after N consecutive
frames report a confident match at a spatially CONSISTENT location (not
jumping around) -- this guards against a single-frame fluke match on this
repetitive terrain. Once accepted, the tracker is freshly RE-INITIALIZED
(tracker.init) at the found location/scale -- its old internal state is
discarded rather than trusted, since the object may have moved
substantially while lost.

While TRACKING, behavior is unchanged from Stage 3: tracker.update() every
frame, score-gated hysteresis with the box-size sanity cap.

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    q / ESC - quit

Usage:
    python stage5_redetection.py --video "../ex/track-train.mp4"
    python stage5_redetection.py --video "../ex/track-train.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2
import numpy as np

from stage1_overlay import OverlayCleaner, pick_point_by_click, VX0, VX1
from stage3_loss_detection import (
    is_valid, crop_template, draw_tracking, draw_lost,
    SCORE_THRESHOLD, LOST_N_FRAMES, RECOVER_N_FRAMES, MAX_BBOX_AREA_FRAC,
    GOOD_TEMPLATE_SCORE,
)

BOX_SIZE = 40
WINDOW_NAME = "Stage 5 - Re-detection"
DEFAULT_MODEL = "models/object_tracking_vittrack_2023sep.onnx"

SEARCH_SCALES = [0.8, 1.2, 1.8]        # relative to the reference template's own size
REDETECT_MATCH_THRESHOLD = 0.55         # min NCC to count as a re-detect candidate
REDETECT_CONSISTENCY_PX = 60            # max center drift between consecutive candidate
                                         # frames to count as "the same" candidate (not a fluke)
REDETECT_RECOVER_N = 2                  # consecutive confident+consistent frames to commit
REDETECT_DOWNSCALE = 2                  # search a half-resolution frame (see note below)
REDETECT_EVERY_N_FRAMES = 2             # only run the search on every Nth LOST frame

# Why still throttled even though the local search window is cheap (~1.4ms):
# removing the throttle entirely was tested and made things WORSE -- 214
# state transitions instead of 36, constantly flickering onto a series of
# different nearby look-alike patches rather than settling. This footage's
# terrain is repetitive even within a small local neighborhood, not just
# globally, so searching every single frame gives the false-match-of-the-
# moment more chances to win before the real object is confirmed. The every-
# other-frame throttle acts as an accidental damping factor here.

# Why downscaled + throttled: profiling showed a full-resolution multi-scale
# search (6 scales x 2 templates x 1440x1080 region) costs ~270ms/frame --
# nowhere near real-time. cv2.matchTemplate's cost scales with the search
# region's pixel count, so searching at half resolution cuts it ~4x; cutting
# to 3 scales and searching only every other LOST frame brings the average
# well under the 30 FPS budget. The tradeoff is a couple of frames' extra
# latency to reacquire and slightly less precise localization (corrected
# quickly once the tracker re-initializes and starts refining every frame).

REDETECT_INITIAL_RADIUS = 150   # px, half-width of the first search window around last_good_bbox
REDETECT_RADIUS_GROWTH = 150    # px, how much the window grows per unsuccessful search attempt

# Why an expanding LOCAL window instead of always searching the whole frame:
# a first full-frame-search version of this stage was tested and found to
# frequently "re-detect" onto the WRONG patch of ground -- this footage's
# terrain is fairly repetitive, so a small appearance template can score a
# deceptively high match (0.7-0.95 NCC) against unrelated look-alike patches
# scattered anywhere in the frame. Measured: 54 state transitions (27 "re-
# detections") on one test run, with recovered box positions jumping by
# hundreds/thousands of pixels between consecutive recoveries -- a real
# object cannot teleport across the frame in a few frames, so most of these
# were false locks, not real re-acquisitions. Restricting the search to a
# small area around the last known position first, and only expanding it the
# longer the object stays missing, sharply cuts the false-positive
# opportunities while still covering the whole frame eventually if the
# object genuinely reappears far from where it was lost.


def multi_scale_search(gray_region, templates, scales, downscale=REDETECT_DOWNSCALE):
    """Search `gray_region` (searched at 1/downscale resolution for speed)
    for the best match against any of `templates` at any of `scales`
    (relative to each template's own size). Returns (score, bbox) in
    gray_region's FULL-resolution coordinates, or (-1.0, None)."""
    small_region = cv2.resize(gray_region, None, fx=1 / downscale, fy=1 / downscale,
                               interpolation=cv2.INTER_AREA)
    best_score = -1.0
    best_bbox = None
    for tmpl in templates:
        if tmpl is None:
            continue
        base_h, base_w = tmpl.shape[:2]
        for s in scales:
            tw = max(4, int(round(base_w * s / downscale)))
            th = max(4, int(round(base_h * s / downscale)))
            if tw >= small_region.shape[1] or th >= small_region.shape[0]:
                continue
            resized = cv2.resize(tmpl, (tw, th))
            result = cv2.matchTemplate(small_region, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_bbox = (max_loc[0] * downscale, max_loc[1] * downscale,
                             tw * downscale, th * downscale)
    return best_score, best_bbox


def main():
    parser = argparse.ArgumentParser(description="Stage 5: active re-detection search while LOST")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--x", type=int, default=None, help="Manual pixel x on frame 1")
    parser.add_argument("--y", type=int, default=None, help="Manual pixel y on frame 1")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to VitTrack ONNX model")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--lost-n", type=int, default=LOST_N_FRAMES)
    parser.add_argument("--recover-n", type=int, default=RECOVER_N_FRAMES)
    parser.add_argument("--max-bbox-area-frac", type=float, default=MAX_BBOX_AREA_FRAC)
    parser.add_argument("--redetect-threshold", type=float, default=REDETECT_MATCH_THRESHOLD)
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
    redetect_count = 0
    search_radius_attempts = 0
    last_good_bbox = init_box
    last_candidate_center = None
    score = 1.0
    bbox = init_box
    n_lost_frames = 0
    n_transitions = 0

    while True:
        base = cleaned if show_cleaned else frame
        display = base.copy()

        if show_mask:
            display[cleaner.full_mask > 0] = (0, 0, 255)

        if state == "TRACKING":
            draw_tracking(display, bbox, score)
        else:
            draw_lost(display, last_good_bbox, score)
            n_lost_frames += 1

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

        if state == "TRACKING":
            _, bbox = tracker.update(cleaned)
            score = tracker.getTrackingScore()
            valid = is_valid(bbox, score, frame_area)
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
                    redetect_count = 0
                    last_candidate_center = None
                    search_radius_attempts = 0
                    n_transitions += 1
                    print(f"Frame {frame_idx} | TRACKING->LOST | score={score:.3f} | bbox={bbox}")
        else:  # LOST -- active search, tracker.update() not consulted
            if n_lost_frames % REDETECT_EVERY_N_FRAMES != 0:
                fps_frame_count += 1
                now = time.perf_counter()
                if now - fps_window_start >= 0.5:
                    display_fps = fps_frame_count / (now - fps_window_start)
                    fps_frame_count = 0
                    fps_window_start = now
                continue

            gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
            lx, ly, lw, lh = last_good_bbox
            center_x, center_y = lx + lw / 2, ly + lh / 2
            radius = REDETECT_INITIAL_RADIUS + REDETECT_RADIUS_GROWTH * search_radius_attempts
            rx0 = int(max(VX0, center_x - radius))
            ry0 = int(max(0, center_y - radius))
            rx1 = int(min(VX1, center_x + radius))
            ry1 = int(min(height, center_y + radius))
            search_region = gray[ry0:ry1, rx0:rx1]

            sim, region_bbox = (-1.0, None)
            if search_region.shape[0] > 8 and search_region.shape[1] > 8:
                sim, region_bbox = multi_scale_search(search_region, [orig_template, recent_template], SEARCH_SCALES)
            score = sim  # for on-screen "stale" readout

            if region_bbox is not None and sim >= args.redetect_threshold:
                rrx, rry, rw, rh = region_bbox
                candidate_bbox = (rrx + rx0, rry + ry0, rw, rh)
                cx, cy = rrx + rx0 + rw / 2, rry + ry0 + rh / 2

                consistent = (
                    last_candidate_center is not None
                    and abs(cx - last_candidate_center[0]) <= REDETECT_CONSISTENCY_PX
                    and abs(cy - last_candidate_center[1]) <= REDETECT_CONSISTENCY_PX
                )
                last_candidate_center = (cx, cy)

                if consistent:
                    redetect_count += 1
                else:
                    redetect_count = 1
                search_radius_attempts = 0

                if redetect_count >= REDETECT_RECOVER_N:
                    tracker = cv2.TrackerVit_create(params)
                    tracker.init(cleaned, candidate_bbox)
                    state = "TRACKING"
                    bad_count = 0
                    bbox = candidate_bbox
                    last_good_bbox = candidate_bbox
                    refreshed = crop_template(cleaned, candidate_bbox, template_size)
                    if refreshed is not None:
                        recent_template = refreshed
                    n_transitions += 1
                    print(f"Frame {frame_idx} | LOST->TRACKING (re-detected) | sim={sim:.3f} | bbox={candidate_bbox} | radius={radius}")
            else:
                redetect_count = 0
                last_candidate_center = None
                search_radius_attempts += 1

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            display_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_window_start = now

    print(f"Summary: {n_transitions} state transitions, {n_lost_frames}/{frame_idx} frames displayed as LOST")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
