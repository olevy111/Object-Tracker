"""
Stage 4.5 + Stage 5 -- Delayed init near the crosshair, and re-detection
(ORB feature matching + static-region guard + velocity-predicted two-tier search).

Stage 4.5 (delayed initialization): the plan flagged this as a conditional,
last-resort stage, needed only if data showed tiny/fragile objects near the
center crosshair couldn't be locked onto directly. Real testing surfaced
exactly that: VitTrack's own confidence score was noisy enough right at the
crosshair (three bad frames within the first 4) to trip LOST almost
immediately for a center click, even though the identical hysteresis logic
held fine for an off-center click of the same object. Fix: if the click
lands within NEAR_CROSSHAIR_RADIUS of the fixed crosshair center, the
tracker is NOT initialized yet. Instead the clicked ground point is HELD
until the drone's motion carries it far enough from the crosshair to sit
on clean, unobstructed ground (or a MAX_HOLD_FRAMES safety cap is hit so
it never waits forever). Only then is a fresh template grabbed and the
tracker initialized normally, exactly as if the click had landed on clear
ground to begin with. A click that's already far from the crosshair skips
all of this and initializes immediately, as before.

How the held point is updated (revised after a real placement-accuracy bug):
the first version updated the point every frame by chaining the Stage 4 GMC
per-frame affine estimate (point = transform(point, M), frame after frame).
Measured against where the real object actually was (using VitTrack's own
early trajectory as ground truth), this drifted increasingly with hold
duration -- 2-10px for a hold that finished quickly, but 17-26px by ~60
held frames for a slower one, enough to place a 40px box mostly or
entirely off the real object. This is dead-reckoning error: each frame's
small estimation noise compounds, and the aggregate scene-wide affine
model doesn't perfectly capture one specific point's true local motion
(parallax, fit residual) either. Tried and rejected: single-point KLT on
the click pixel itself (worse, up to 70px -- a click point is often not a
strong corner, so KLT can drift/get stuck on it); chaining a local-corner-
cluster's average per-frame motion (no better, same compounding problem).
What worked: track a small cluster of real nearby corners (goodFeaturesToTrack
in a neighborhood around the click, excluding the overlay mask) via a
SINGLE optical-flow call directly from FRAME 1 to the CURRENT frame each
time (recomputed fresh, never chained) -- this avoids accumulating each
frame's noise, since it's always one measurement, not N composed ones. It
still degrades on very long holds (large frame-to-frame appearance change
breaks single-shot matching too, ~30-40px by frame 60+), so
NEAR_CROSSHAIR_RADIUS was also reduced (80px -> 45px) so the typical hold
finishes before that degradation zone, not after it.


Builds on Stage 3 (score-gated loss detection + hysteresis). Replaces the
passive "keep calling tracker.update() and hope it stumbles back onto the
object" behavior of Stages 3/4 with an ACTIVE search while LOST, modeled on
a previously-working design for this exact repetitive-desert footage.

Why ORB instead of cv2.matchTemplate (this replaces a first version of this
stage that used matchTemplate): plain patch correlation floods with false
positives on this footage's repetitive terrain -- measured, a matchTemplate
version produced re-detected boxes jumping by hundreds/thousands of pixels
between consecutive "recoveries", clearly locking onto look-alike ground.
Distinctive ORB keypoint + descriptor matching (BFMatcher + knnMatch +
Lowe's ratio test, requiring a minimum number of ratio-passing matches) is
more discriminative because it requires a cluster of individually-
distinctive local features to agree, not just one correlation peak.
Calibration note: ORB's default edgeThreshold/patchSize (31) leaves NO
valid detection area on a small (~40px) template -- it produced ZERO
keypoints. Both are lowered to 7, and templates are cropped with extra
padding for keypoint richness.

Two-tier search, SEARCHING_LOCAL vs SEARCHING_GLOBAL:
    - SEARCHING_LOCAL runs every LOST frame (cheap -- ORB only runs inside
      a small ROI). The ROI is centered not on the stale last-known
      position, but on a PREDICTED position: linear extrapolation of the
      object's velocity (from its last 3 tracked positions) times how many
      frames it's been lost -- if the object/camera has real motion, the
      search center moves with it instead of searching where the object
      USED to be. The ROI radius grows with elapsed lost-frame count
      (not with failed attempts), capped at a max.
    - After LOCAL_SEARCH_FRAMES frames without success, escalates to
      SEARCHING_GLOBAL: the whole video-content region, throttled to once
      every GLOBAL_SEARCH_INTERVAL frames to stay within the FPS budget
      (full-frame ORB is more expensive than a small ROI).
    - The frame is NOT downscaled for either tier -- ORB/BRIEF are only
      mildly scale-tolerant, and downscaling a small object's already-
      sparse keypoints collapses match counts further.

Matching core (shared by both tiers): ORB features are matched against TWO
saved templates -- an original anchor (fixed forever, from the initial
click) and a periodically refreshed one (every ~30 confident-tracking
frames), so a long occlusion or scale change doesn't strand the system on
a stale template while the ground-truth original is never lost.

Static-region rejection guard (critical safety gate): before accepting ANY
candidate, however strong its ORB match, its region is compared against the
SAME region in the previous frame via cv2.absdiff. If the mean difference
is near zero, the candidate is pixel-static and rejected outright -- real
scene content changes frame to frame (even subtly); a screen-locked HUD
element (the timer, the RC/HD/Mbps/battery strip) does not. Calibrated on
this footage: real terrain averages ~4.0 mean absdiff, the on-screen timer
region averages ~1.1 -- a threshold of 2.0 cleanly separates them. Verified
live: without this guard, ORB repeatedly re-locked onto the bottom-right
HUD strip (same position recurring every attempt for 60+ frames straight).

On finding a non-static, sufficiently-matched candidate (either tier), the
tracker is fully re-initialized at that location immediately -- no extra
multi-frame consistency wait on top of the match-count + static-region
gates, matching the reference design this stage is modeled on.

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    q / ESC - quit

Usage:
    python stage5_redetection.py --video "../ex/track-train.mp4"
    python stage5_redetection.py --video "../ex/track-train.mp4" --x 960 --y 540
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

from stage1_overlay import OverlayCleaner, pick_point_by_click, VX0, VX1, CX, CY
from stage3_loss_detection import (
    is_valid, draw_tracking,
    SCORE_THRESHOLD, LOST_N_FRAMES, RECOVER_N_FRAMES, MAX_BBOX_AREA_FRAC,
)

BOX_SIZE = 40
WINDOW_NAME = "Stage 5 - Re-detection (ORB)"
DEFAULT_MODEL = "models/object_tracking_vittrack_2023sep.onnx"

# -- ORB matching --
ORB_NFEATURES = 500
ORB_EDGE_THRESHOLD = 7   # default (31) leaves no valid area on a ~40-90px template -- see module docstring
ORB_PATCH_SIZE = 7
ORB_RATIO_TEST = 0.75
ORB_MIN_MATCHES = 10
ORB_TEMPLATE_PADDING = 25     # extra context (px) around the tracking box when cropping an ORB template;
                              # a bare 40px box yields ~0-5 keypoints, not enough to ever reach ORB_MIN_MATCHES
TEMPLATE_REFRESH_FRAMES = 30  # refresh the "recent" template every N confident TRACKING frames

# -- Re-detection probation (protects the original object's identity) --
CONFIRM_FRAMES = 20
# Why: ORB match count + the static-region guard aren't enough to guarantee a
# re-detection is the SAME object on this repetitive terrain (measured: even
# RANSAC geometric verification of the matched keypoints didn't cleanly
# separate correct from unrelated regions -- see module docstring). The real
# risk isn't a few wrong-looking frames; it's that a wrong candidate used to
# get trusted IMMEDIATELY, overwriting last_good_bbox/pos_history/recent_orb
# -- the system's only memory of where and what the real object was. From
# then on, every subsequent search was centered on the wrong place and
# matched against the wrong appearance: the real object's "identity" was
# gone. Now a re-detected candidate starts on PROBATION: the tracker runs on
# it, but last_good_bbox/pos_history/recent_orb are left untouched until it
# survives CONFIRM_FRAMES consecutive valid frames. If it fails first
# (common for false locks, which tend to collapse within a handful of
# frames), it's discarded and the NEXT search resumes from the last
# CONFIRMED position/appearance -- never from the failed guess.

# -- Static-region rejection guard --
STATIC_DIFF_THRESHOLD = 2.0
# Calibrated on this footage: the on-screen timer region averages ~1.1 mean
# absdiff frame-to-frame (98% of frames below 2.0), while real terrain
# averages ~4.0 (only 18% falsely below 2.0). See module docstring.

# -- Stage 4.5: delayed init near the crosshair --
NEAR_CROSSHAIR_RADIUS = 45   # px; a click within this distance of the crosshair center delays init
                              # (reduced from an initial 80 -- see module docstring: a smaller radius
                              # means a shorter hold, finishing before single-shot LK match quality
                              # degrades over long holds)
HOLD_CORNER_NEIGHBORHOOD = 70  # px radius around the click to search for trackable local corners
HOLD_MAX_CORNERS = 25
MAX_HOLD_FRAMES = 90         # safety cap (~1.5s @ 60fps) so holding never waits forever if the
                              # camera barely moves or GMC repeatedly can't estimate motion

# -- Two-tier search geometry --
POS_HISTORY_LEN = 5              # tracked positions kept for velocity estimation
LOCAL_SEARCH_BASE_RADIUS = 150   # px, local ROI half-width at the moment loss is declared
LOCAL_SEARCH_RADIUS_GROWTH = 8   # px, ROI growth per elapsed LOST frame
LOCAL_SEARCH_MAX_RADIUS = 500    # px, ROI half-width cap
LOCAL_SEARCH_FRAMES = 45         # frames of local search before escalating to global
GLOBAL_SEARCH_INTERVAL = 4       # throttle: run the global tier once every N LOST frames


def make_orb():
    return cv2.ORB_create(nfeatures=ORB_NFEATURES, edgeThreshold=ORB_EDGE_THRESHOLD,
                           patchSize=ORB_PATCH_SIZE)


def crop_orb_template(frame, bbox, padding=ORB_TEMPLATE_PADDING):
    """Crop `bbox` plus extra context out of `frame` (clipped to bounds) and
    return it as a grayscale patch."""
    x, y, w, h = [int(round(v)) for v in bbox]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(frame.shape[1], x + w + padding)
    y1 = min(frame.shape[0], y + h + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def orb_locate(orb, bf, search_gray, template_entries, box_size, offset=(0, 0)):
    """Search `search_gray` for the best match against any of
    `template_entries` (list of (keypoints, descriptors) from
    orb.detectAndCompute on reference templates). Returns (bbox, n_matches)
    in FULL-frame coordinates (using `offset` = the search region's
    top-left in the full frame), or (None, 0). `box_size` = (w, h) of the
    tracking box to place at the matched location -- NOT derived from the
    spread of matched keypoints (tried and found to blow up unboundedly:
    weak/generic matches scatter across a wide area, and feeding that
    inflated size back into the next template crop compounds into a
    runaway growing box, the same failure shape as the Stage 2 degenerate
    full-frame lock)."""
    kp2, des2 = orb.detectAndCompute(search_gray, None)
    if des2 is None or len(des2) < 2:
        return None, 0

    best_bbox = None
    best_count = 0
    bw, bh = box_size
    for kp1, des1 in template_entries:
        if des1 is None or len(des1) < 2:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < ORB_RATIO_TEST * n.distance]
        if len(good) < ORB_MIN_MATCHES or len(good) <= best_count:
            continue

        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Median, not mean: robust against a handful of scattered outlier
        # matches pulling the estimated center away from the main cluster.
        cx = float(np.median(dst_pts[:, 0, 0]))
        cy = float(np.median(dst_pts[:, 0, 1]))
        bbox = (float(cx - bw / 2 + offset[0]), float(cy - bh / 2 + offset[1]), float(bw), float(bh))
        best_bbox = bbox
        best_count = len(good)

    return best_bbox, best_count


def is_static_region(curr_frame, prev_frame, bbox, threshold=STATIC_DIFF_THRESHOLD):
    """True if `bbox`'s content is (near-)identical between frames -- a
    screen-locked overlay graphic, not real scene content."""
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(curr_frame.shape[1], x + w), min(curr_frame.shape[0], y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return True
    curr_region = curr_frame[y0:y1, x0:x1]
    prev_region = prev_frame[y0:y1, x0:x1]
    return float(cv2.absdiff(curr_region, prev_region).mean()) < threshold


def estimate_velocity(pos_history):
    """Velocity (px/frame) from the first and last of the last 3 tracked
    positions. Returns (0, 0) until enough history has accumulated."""
    if len(pos_history) < 3:
        return 0.0, 0.0
    (x0, y0), (x1, y1) = pos_history[-3], pos_history[-1]
    return (x1 - x0) / 2.0, (y1 - y0) / 2.0


def predicted_center(pos_history, lost_frames):
    """Linear extrapolation of the last known position by `lost_frames`
    steps of the estimated velocity -- keeps the local search ROI centered
    on where the object should be NOW, not where it was last seen."""
    cx, cy = pos_history[-1]
    vx, vy = estimate_velocity(pos_history)
    return cx + vx * lost_frames, cy + vy * lost_frames


def find_hold_corners(gray1, point, overlay_mask, neighborhood=HOLD_CORNER_NEIGHBORHOOD,
                       max_corners=HOLD_MAX_CORNERS):
    """Find real trackable corners in a small neighborhood around the click,
    excluding the overlay mask (fixed crosshair/X-lines have zero true
    motion and would corrupt the estimate)."""
    x0 = max(0, point[0] - neighborhood)
    y0 = max(0, point[1] - neighborhood)
    x1 = min(gray1.shape[1], point[0] + neighborhood)
    y1 = min(gray1.shape[0], point[1] + neighborhood)
    roi_mask = np.zeros(gray1.shape, np.uint8)
    roi_mask[y0:y1, x0:x1] = 255
    roi_mask[overlay_mask > 0] = 0
    return cv2.goodFeaturesToTrack(gray1, maxCorners=max_corners, qualityLevel=0.01,
                                    minDistance=5, mask=roi_mask)


def estimate_held_point(gray1, curr_gray, corners0, click_point):
    """Single-shot optical flow directly from frame 1 to the CURRENT frame
    (never chained frame-to-frame -- see module docstring on why chaining
    was tried and rejected). Returns the click point offset by the median
    motion of the surviving corners, or None if too few survive to trust."""
    if corners0 is None or len(corners0) < 3:
        return None
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray1, curr_gray, corners0, None,
                                                   winSize=(31, 31), maxLevel=4)
    status = status.reshape(-1).astype(bool)
    good_old = corners0[status]
    good_new = new_pts[status]
    if len(good_old) < 3:
        return None
    motion = np.median((good_new - good_old).reshape(-1, 2), axis=0)
    return float(click_point[0] + motion[0]), float(click_point[1] + motion[1])


def draw_holding(frame, held_point, hold_frames, dist_to_crosshair, color=(0, 200, 255)):
    x, y = int(held_point[0]), int(held_point[1])
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
    cv2.putText(frame, f"HOLDING (near crosshair) frame={hold_frames} dist={dist_to_crosshair:.0f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def draw_lost_orb(frame, last_good_bbox, n_matches, mode, color=(0, 0, 255), dim_color=(120, 120, 120)):
    if last_good_bbox is not None:
        x, y, w, h = [int(v) for v in last_good_bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), dim_color, 1)
    cv2.putText(frame, f"LOST [{mode}]  orb_matches={n_matches}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def draw_probation(frame, bbox, score, probation_count, color=(0, 165, 255)):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"TENTATIVE score={score:.2f} confirm {probation_count}/{CONFIRM_FRAMES}",
                (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    global ORB_MIN_MATCHES
    parser = argparse.ArgumentParser(description="Stage 5: ORB-based re-detection with static-region guard")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--x", type=int, default=None, help="Manual pixel x on frame 1")
    parser.add_argument("--y", type=int, default=None, help="Manual pixel y on frame 1")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to VitTrack ONNX model")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--lost-n", type=int, default=LOST_N_FRAMES)
    parser.add_argument("--recover-n", type=int, default=RECOVER_N_FRAMES)
    parser.add_argument("--max-bbox-area-frac", type=float, default=MAX_BBOX_AREA_FRAC)
    parser.add_argument("--min-orb-matches", type=int, default=ORB_MIN_MATCHES)
    parser.add_argument("--near-crosshair-radius", type=float, default=NEAR_CROSSHAIR_RADIUS)
    parser.add_argument("--no-delayed-init", action="store_true",
                         help="Disable Stage 4.5 delayed init; always initialize immediately (for comparison)")
    parser.add_argument("--show-cleaned", action="store_true")
    parser.add_argument("--show-mask", action="store_true")
    args = parser.parse_args()
    ORB_MIN_MATCHES = args.min_orb_matches

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
    params = cv2.TrackerVit_Params()
    params.net = args.model
    orb = make_orb()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    cv2.namedWindow(WINDOW_NAME)
    show_cleaned = args.show_cleaned
    show_mask = args.show_mask

    frame_idx = 0
    fps_window_start = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    cleaned = cleaned1
    prev_cleaned = cleaned1
    frame = frame1

    bad_count = 0
    frames_since_refresh = 0
    lost_frames = 0
    match_count_display = 0
    search_mode_display = "-"
    n_lost_frames = 0
    n_transitions = 0
    n_static_rejections = 0

    tracker = None
    orig_orb = (None, None)
    recent_orb = (None, None)

    dist_to_crosshair = math.hypot(point[0] - CX, point[1] - CY)
    if args.no_delayed_init or dist_to_crosshair > args.near_crosshair_radius:
        init_box = (point[0] - half, point[1] - half, args.box_size, args.box_size)
        tracker = cv2.TrackerVit_create(params)
        tracker.init(cleaned1, init_box)
        print(f"Tracker initialized on cleaned frame, box={init_box}")

        orig_tmpl_gray = crop_orb_template(cleaned1, init_box)
        orig_orb = orb.detectAndCompute(orig_tmpl_gray, None) if orig_tmpl_gray is not None else (None, None)
        recent_orb = orig_orb
        print(f"Original ORB template: {len(orig_orb[0]) if orig_orb[0] else 0} keypoints")

        state = "TRACKING"
        probation = False
        probation_count = 0
        pos_history = [(init_box[0] + init_box[2] / 2, init_box[1] + init_box[3] / 2)]
        last_good_bbox = init_box
        bbox = init_box
        score = 1.0
        held_point = None
        hold_frames = 0
        gray1_hold = None
        hold_corners0 = None
    else:
        # Stage 4.5: click is too close to the crosshair to trust an immediate
        # init (see module docstring) -- hold the ground point and update it
        # each frame via single-shot optical flow from frame 1 on a small
        # cluster of real nearby corners, until it clears the crosshair.
        print(f"Click is {dist_to_crosshair:.0f}px from the crosshair (<= {args.near_crosshair_radius:.0f}) "
              f"-- holding until it clears the crosshair before initializing.")
        state = "HOLDING"
        held_point = (float(point[0]), float(point[1]))
        hold_frames = 0
        gray1_hold = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hold_corners0 = find_hold_corners(gray1_hold, point, cleaner.full_mask)
        print(f"Holding with {len(hold_corners0) if hold_corners0 is not None else 0} local reference corners")
        pos_history = [held_point]
        last_good_bbox = None
        bbox = None
        score = 0.0

    while True:
        base = cleaned if show_cleaned else frame
        display = base.copy()

        if show_mask:
            display[cleaner.full_mask > 0] = (0, 0, 255)

        if state == "TRACKING":
            if probation:
                draw_probation(display, bbox, score, probation_count)
            else:
                draw_tracking(display, bbox, score)
        elif state == "HOLDING":
            draw_holding(display, held_point, hold_frames, math.hypot(held_point[0] - CX, held_point[1] - CY))
        else:
            draw_lost_orb(display, last_good_bbox, match_count_display, search_mode_display)
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

        prev_cleaned = cleaned
        cleaned = cleaner.clean(frame)

        if state == "HOLDING":
            curr_gray_hold = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            estimated = estimate_held_point(gray1_hold, curr_gray_hold, hold_corners0, point)
            if estimated is not None:
                held_point = estimated
            hold_frames += 1

            dist_to_crosshair = math.hypot(held_point[0] - CX, held_point[1] - CY)
            if dist_to_crosshair > args.near_crosshair_radius or hold_frames >= MAX_HOLD_FRAMES:
                hx = min(max(held_point[0], VX0 + half), VX1 - half)
                hy = min(max(held_point[1], half), height - half)
                init_box = (int(round(hx - half)), int(round(hy - half)), args.box_size, args.box_size)

                tracker = cv2.TrackerVit_create(params)
                tracker.init(cleaned, init_box)
                orig_tmpl_gray = crop_orb_template(cleaned, init_box)
                orig_orb = orb.detectAndCompute(orig_tmpl_gray, None) if orig_tmpl_gray is not None else (None, None)
                recent_orb = orig_orb

                state = "TRACKING"
                probation = False
                probation_count = 0
                pos_history = [(hx, hy)]
                last_good_bbox = init_box
                bbox = init_box
                score = 1.0
                reason = "cleared crosshair" if dist_to_crosshair > args.near_crosshair_radius else "max hold reached"
                print(f"Frame {frame_idx} | HOLDING->TRACKING ({reason}) | "
                      f"init_box={init_box} after {hold_frames} held frames, ORB kp={len(orig_orb[0]) if orig_orb[0] else 0}")

        elif state == "TRACKING":
            _, bbox = tracker.update(cleaned)
            score = tracker.getTrackingScore()
            valid = is_valid(bbox, score, frame_area)

            if probation:
                lost_frames += 1  # still mid lost-episode until confirmed -- keep the search clock running
                if valid:
                    bad_count = 0
                    probation_count += 1
                    if probation_count >= CONFIRM_FRAMES:
                        probation = False
                        last_good_bbox = bbox
                        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                        pos_history = [(cx, cy)]
                        refreshed_gray = crop_orb_template(cleaned, bbox)
                        if refreshed_gray is not None:
                            recent_orb = orb.detectAndCompute(refreshed_gray, None)
                        frames_since_refresh = 0
                        print(f"Frame {frame_idx} | TENTATIVE->TRACKING (confirmed) | bbox={bbox}")
                else:
                    bad_count += 1
                    probation_count = 0
                    if bad_count >= args.lost_n:
                        state = "LOST"
                        probation = False
                        n_transitions += 1
                        print(f"Frame {frame_idx} | TENTATIVE->LOST (candidate did not hold) | "
                              f"score={score:.3f} | resuming search from last CONFIRMED position")
            elif valid:
                bad_count = 0
                last_good_bbox = bbox
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                pos_history.append((cx, cy))
                if len(pos_history) > POS_HISTORY_LEN:
                    pos_history.pop(0)
                frames_since_refresh += 1
                if frames_since_refresh >= TEMPLATE_REFRESH_FRAMES:
                    refreshed_gray = crop_orb_template(cleaned, bbox)
                    if refreshed_gray is not None:
                        recent_orb = orb.detectAndCompute(refreshed_gray, None)
                    frames_since_refresh = 0
            else:
                bad_count += 1
                if bad_count >= args.lost_n:
                    state = "LOST"
                    lost_frames = 0
                    n_transitions += 1
                    print(f"Frame {frame_idx} | TRACKING->LOST | score={score:.3f} | bbox={bbox}")
        else:  # LOST -- active ORB search, tracker.update() not consulted
            lost_frames += 1
            use_global = lost_frames > LOCAL_SEARCH_FRAMES
            search_mode_display = "GLOBAL" if use_global else "LOCAL"

            run_this_frame = True
            if use_global and lost_frames % GLOBAL_SEARCH_INTERVAL != 0:
                run_this_frame = False

            candidate_bbox = None
            n_matches = 0
            if run_this_frame:
                gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
                if use_global:
                    rx0, ry0, rx1, ry1 = VX0, 0, VX1, height
                else:
                    radius = min(LOCAL_SEARCH_BASE_RADIUS + lost_frames * LOCAL_SEARCH_RADIUS_GROWTH,
                                 LOCAL_SEARCH_MAX_RADIUS)
                    pcx, pcy = predicted_center(pos_history, lost_frames)
                    rx0 = int(max(VX0, pcx - radius))
                    ry0 = int(max(0, pcy - radius))
                    rx1 = int(min(VX1, pcx + radius))
                    ry1 = int(min(height, pcy + radius))
                search_region = gray[ry0:ry1, rx0:rx1]

                if search_region.shape[0] > 8 and search_region.shape[1] > 8:
                    candidate_bbox, n_matches = orb_locate(
                        orb, bf, search_region, [orig_orb, recent_orb],
                        box_size=(args.box_size, args.box_size), offset=(rx0, ry0))

            match_count_display = n_matches

            if candidate_bbox is not None:
                if is_static_region(cleaned, prev_cleaned, candidate_bbox):
                    n_static_rejections += 1
                    print(f"Frame {frame_idx} | LOST (static-region reject) | matches={n_matches} | "
                          f"bbox={tuple(round(v, 1) for v in candidate_bbox)}")
                else:
                    # Enter PROBATION -- do NOT touch last_good_bbox / pos_history /
                    # recent_orb yet. If this candidate is wrong, the next search
                    # attempt must still start from the last CONFIRMED position and
                    # appearance, not from this guess (see CONFIRM_FRAMES docstring).
                    tracker = cv2.TrackerVit_create(params)
                    tracker.init(cleaned, tuple(int(round(v)) for v in candidate_bbox))
                    state = "TRACKING"
                    probation = True
                    probation_count = 0
                    bad_count = 0
                    bbox = candidate_bbox
                    n_transitions += 1
                    print(f"Frame {frame_idx} | LOST->TENTATIVE (re-detected, unconfirmed) | matches={n_matches} | "
                          f"bbox={tuple(round(v, 1) for v in candidate_bbox)} | mode={search_mode_display}")

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            display_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_window_start = now

    print(f"Summary: {n_transitions} state transitions, {n_lost_frames}/{frame_idx} frames displayed as LOST, "
          f"{n_static_rejections} static-region rejections")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
