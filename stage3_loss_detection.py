"""
Stage 3 -- Loss detection (score-gated + hysteresis) + appearance-verified recovery.

Builds on Stage 2. Declares a TRACKING / LOST state using the confidence
score (NOT the tracker's own `ok` flag, which Stage 2 showed is unreliable)
with hysteresis so the state doesn't flicker:
    - N consecutive bad frames (score below threshold, or a bogus box) ->
      TRACKING -> LOST
    - M consecutive GOOD-AND-MATCHING frames while LOST -> LOST -> TRACKING

Why "bogus box" is also checked (not just the score threshold): Stage 2
testing found that after an extended loss, TrackerVit can lock onto a
degenerate box covering almost the entire frame while reporting a
deceptively HIGH score (~0.8). Score-gating alone would never catch this
(0.8 is above any reasonable threshold), so a box-size sanity bound is
included as a second, independent gate -- either one failing counts the
frame as "bad" for the hysteresis counters.

Why appearance verification was added (real bug found while testing this
stage): score + box-size sanity alone are not enough to trust a recovery.
TrackerVit keeps running every frame even while displayed as LOST, and on
this repetitive open-desert footage its fixed template can latch onto a
different lookalike patch of ground and report a perfectly plausible score
and box size for the WRONG object. Neither gate catches that, because
both are about "does this look like a trackable thing", not "is this the
SAME thing we started with". The fix: before accepting a LOST -> TRACKING
transition, the candidate box's cleaned-frame content is compared via
normalized cross-correlation (cv2.matchTemplate) against saved reference
templates (the original click template, plus the most recent
high-confidence template so gradual scale/rotation changes aren't
mistaken for a mismatch). Only if that similarity also clears a threshold
is the recovery accepted -- otherwise the state stays LOST even though the
score/size gates alone would have accepted it.

No active search yet (that's Stage 5) -- once LOST, this stage keeps
feeding frames to the tracker unconditionally and just watches whether it
self-recovers onto something that also matches the saved appearance; it
does not actively scan the frame for the object.

State transitions are logged to console. While LOST, the live (untrustworthy)
box/score are marked "(stale)" and not drawn as if valid; the last known
good box is shown dimmed instead.

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    q / ESC - quit

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

SCORE_THRESHOLD = 0.30        # below this score, frame counts as "bad"
LOST_N_FRAMES = 3             # consecutive bad frames -> TRACKING => LOST
RECOVER_N_FRAMES = 2          # consecutive good+matching frames -> LOST => TRACKING
MAX_BBOX_AREA_FRAC = 0.03     # sanity cap: box bigger than this fraction of the frame is bogus

# Why 0.03 and not the Stage-3-v1 0.08: measured on the real clip, legitimate
# (confirmed correct) boxes topped out around 1.8% of frame area even during
# real scale growth, while the degenerate false-lock reliably exceeded 7%.
# 3% sits with margin in the gap between the two -- this is the dominant,
# reliable gate against the catastrophic full-frame lock.

MAX_BBOX_ASPECT_RATIO = 2.0   # sanity cap: reject a box whose long side is more than this many
                              # times its short side

# Why: found live on a small, low-contrast object sitting next to a diagonal
# road/path -- VitTrack's regression drifted to follow the linear road
# feature instead of staying on the object, growing the box's HEIGHT
# steadily (40 -> 57 -> 81 -> 98 -> 110px) while width stayed contained
# (35-52px) -- an elongated strip straddling the road, confirmed visually,
# not a tight box on the object. This is the same regression-runaway
# instability as the full-frame catastrophic lock (MAX_BBOX_AREA_FRAC above),
# just shaped differently (elongated, not large-area) because a nearby
# linear feature rather than open terrain is what it latched onto. A real
# small object is expected to be roughly compact, not a long thin strip.

TEMPLATE_MATCH_THRESHOLD = 0.10  # min normalized cross-correlation to accept a recovery
GOOD_TEMPLATE_SCORE = 0.5        # refresh the "recent" reference template only on strong frames

# Why appearance matching is a LOOSE secondary gate, not the primary defense:
# measured on the real clip, plain fixed-size-patch normalized cross-correlation
# does not cleanly separate legitimate (confirmed correct) recoveries from the
# degenerate false-lock -- both distributions overlap in the 0.0-0.2 range,
# because this object is tiny, low-texture, and its scale/rotation drifts
# frame to frame. A strict threshold (e.g. 0.5) rejected genuine recoveries
# for the rest of the clip. 0.10 mostly stays out of the way of legitimate
# recoveries while still rejecting clearly anti-correlated (actively
# dissimilar) content. A robust same-size wrong-object rejection needs a
# better descriptor than plain NCC -- left for Stage 5's dedicated
# re-detection design, which can also use the GMC-predicted location (Stage 4)
# to know WHERE to expect the object instead of trusting wherever the
# tracker's own passive update drifted to.


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
    """Crop `bbox` out of `frame`, clipped to frame bounds, and resize to a
    canonical (size, size) grayscale patch so templates captured from boxes
    of different sizes remain directly comparable."""
    x, y, w, h = [int(v) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size))


def template_similarity(frame, bbox, templates, size):
    """Max normalized cross-correlation between the candidate box's content
    and any of the saved reference templates. Returns -1.0 if the box can't
    be cropped at all (fully off-frame)."""
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
