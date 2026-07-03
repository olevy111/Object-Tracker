"""
Stage 4 -- Global Motion Compensation (GMC).

Builds on Stage 3. The drone's camera motion between consecutive frames is
estimated (sparse optical flow + RANSAC affine), and the CURRENT cleaned
frame is warped back into the PREVIOUS frame's coordinate system before the
tracker ever sees it. From the tracker's point of view, the camera is now
stationary frame-to-frame -- its own last-known position stays valid instead
of being invalidated every time the drone pans/rolls, which Stage 2/3 showed
was the dominant cause of tracking loss.

How the motion is estimated and used:
    1. cv2.goodFeaturesToTrack on the PREVIOUS frame, restricted to the real
       video content region and EXCLUDING the overlay mask (crosshair + X
       diagonals) -- required, because those are fixed to the screen, not
       the scene, and would otherwise vote for "zero motion" and corrupt the
       estimate.
    2. cv2.calcOpticalFlowPyrLK tracks those points into the CURRENT frame.
    3. cv2.estimateAffinePartial2D (RANSAC) fits a similarity transform M
       (translation + rotation + uniform scale) mapping previous-frame
       positions to current-frame positions from the matched point pairs.
    4. The current cleaned frame is warped with WARP_INVERSE_MAP so the
       result is expressed in the PREVIOUS frame's coordinate system --
       this is what's fed to tracker.update(), so its internal last-known
       box position (also in that coordinate system) is still where it
       expects it to be, motion aside.
    5. The box the tracker returns (in previous-frame coordinates) is
       mapped forward through M to get the true current-frame box used for
       display, the state machine, and appearance verification.
    Each frame's M is estimated fresh from the true previous/current frames
    (not accumulated across many frames), so estimation noise doesn't
    compound into long-term drift.
    If too few flow matches survive (occlusion, blank sky, RANSAC failure),
    M is None and this frame silently falls back to Stage 3's un-warped
    behavior instead of crashing.

Everything else (score-gated loss detection, hysteresis, box-size sanity,
appearance-verified recovery) is unchanged from Stage 3 and reused directly.

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    g  - toggle an on-screen readout of the estimated motion (dx, dy, matches)
    q / ESC - quit

Usage:
    python stage4_gmc.py --video "../ex/track-train.mp4"
    python stage4_gmc.py --video "../ex/track-train.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2
import numpy as np

from stage1_overlay import OverlayCleaner, pick_point_by_click, VX0, VX1
from stage3_loss_detection import (
    is_valid, crop_template, template_similarity, draw_tracking, draw_lost,
    SCORE_THRESHOLD, LOST_N_FRAMES, RECOVER_N_FRAMES, MAX_BBOX_AREA_FRAC,
    TEMPLATE_MATCH_THRESHOLD, GOOD_TEMPLATE_SCORE,
)

BOX_SIZE = 40
WINDOW_NAME = "Stage 4 - Global Motion Compensation"
DEFAULT_MODEL = "models/object_tracking_vittrack_2023sep.onnx"

GMC_MAX_CORNERS = 300
GMC_QUALITY_LEVEL = 0.01
GMC_MIN_DISTANCE = 10
GMC_MIN_MATCHES = 10          # below this many good flow matches, distrust the estimate
GMC_RANSAC_THRESH = 3.0
GMC_FEATURE_MASK_DILATE = 5   # extra safety margin excluded around the overlay mask
GMC_DOWNSCALE = 4             # feature detection + flow run at 1/4 resolution (see note below)

# Why downscaled: profiling showed cv2.goodFeaturesToTrack alone costs ~48ms
# on the full 1920x1080 frame -- on its own already under 30 FPS. Global
# motion is a low-frequency signal (the whole scene shifting together), so
# it doesn't need full-resolution corners to estimate accurately. Running
# feature detection + optical flow at 1/4 resolution (480x270) cuts that to
# ~3ms combined, a ~13x speedup, before scaling the matched points back up
# to full-resolution coordinates for the affine fit.


def build_flow_feature_mask(width, height, overlay_mask):
    """Valid region for optical-flow feature detection: inside the real
    video content (excludes the black letterbox) and outside the overlay
    mask (crosshair + X diagonals), which is fixed to the screen and would
    otherwise look like "zero motion" scene content and bias the estimate."""
    mask = np.zeros((height, width), np.uint8)
    mask[:, VX0:VX1] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (GMC_FEATURE_MASK_DILATE, GMC_FEATURE_MASK_DILATE))
    excluded = cv2.dilate(overlay_mask, kernel, iterations=1)
    mask[excluded > 0] = 0
    return mask


def estimate_motion(prev_gray, curr_gray, feature_mask_small, scale=GMC_DOWNSCALE):
    """Returns (M, n_matches). M is a 2x3 affine mapping previous-frame
    positions to current-frame positions (in full-resolution coordinates),
    or None if too few reliable matches survive to trust the estimate.
    Feature detection + optical flow run on downscaled frames for speed;
    matched points are scaled back up before the affine fit."""
    small_prev = cv2.resize(prev_gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(curr_gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)

    pts_prev = cv2.goodFeaturesToTrack(small_prev, maxCorners=GMC_MAX_CORNERS,
                                        qualityLevel=GMC_QUALITY_LEVEL,
                                        minDistance=max(1, GMC_MIN_DISTANCE // scale),
                                        mask=feature_mask_small)
    if pts_prev is None or len(pts_prev) < GMC_MIN_MATCHES:
        return None, 0 if pts_prev is None else len(pts_prev)

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(small_prev, small_curr, pts_prev, None)
    status = status.reshape(-1).astype(bool)
    good_prev = pts_prev[status] * scale
    good_curr = pts_curr[status] * scale
    if len(good_prev) < GMC_MIN_MATCHES:
        return None, len(good_prev)

    M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC,
                                        ransacReprojThreshold=GMC_RANSAC_THRESH)
    return M, len(good_prev)


def warp_to_prev(frame, M, size):
    return cv2.warpAffine(frame, M, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)


def transform_bbox(bbox, M):
    x, y, w, h = bbox
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.transform(corners, M).reshape(-1, 2)
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def main():
    parser = argparse.ArgumentParser(description="Stage 4: GMC-compensated tracking with Stage 3 loss detection")
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
    parser.add_argument("--no-gmc", action="store_true", help="Disable GMC (falls back to Stage 3 behavior, for comparison)")
    parser.add_argument("--show-cleaned", action="store_true")
    parser.add_argument("--show-mask", action="store_true")
    parser.add_argument("--show-motion", action="store_true")
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
    flow_mask = build_flow_feature_mask(width, height, cleaner.full_mask)
    flow_mask_small = cv2.resize(flow_mask, None, fx=1 / GMC_DOWNSCALE, fy=1 / GMC_DOWNSCALE,
                                  interpolation=cv2.INTER_NEAREST)

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

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(WINDOW_NAME)

    show_cleaned = args.show_cleaned
    show_mask = args.show_mask
    show_motion = args.show_motion

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
    motion_info = "no motion yet"
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
        if show_motion:
            cv2.putText(display, motion_info, (20, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            show_cleaned = not show_cleaned
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("g"):
            show_motion = not show_motion

        ok, frame = cap.read()
        if not ok:
            print("End of video.")
            break
        frame_idx += 1

        cleaned = cleaner.clean(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        M, n_matches = (None, 0) if args.no_gmc else estimate_motion(prev_gray, curr_gray, flow_mask_small)
        if M is not None:
            search_frame = warp_to_prev(cleaned, M, (width, height))
            dx, dy = M[0, 2], M[1, 2]
            motion_info = f"GMC dx={dx:+.1f} dy={dy:+.1f} matches={n_matches}"
        else:
            search_frame = cleaned
            motion_info = f"GMC unavailable (matches={n_matches})"
        prev_gray = curr_gray

        _, bbox_search = tracker.update(search_frame)
        score = tracker.getTrackingScore()
        bbox = transform_bbox(bbox_search, M) if M is not None else bbox_search
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
                    n_transitions += 1
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
                        n_transitions += 1
                        print(f"Frame {frame_idx} | LOST->TRACKING | score={score:.3f} | sim={sim:.3f} | bbox={bbox}")
                else:
                    good_count = 0
            else:
                good_count = 0

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
