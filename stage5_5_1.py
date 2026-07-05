"""
Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    q / ESC - quit

Usage:
    python stage5_5_1.py --video "../ex/track-train.mp4"
    python stage5_5_1.py --video "../ex/track-train.mp4" --x 960 --y 540
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from stage1_overlay import (
    OverlayCleaner, pick_point_by_click, bbox_overlaps_mask, compute_display_scale, VX0, VX1, CX, CY,
)
from stage3_loss_detection import (
    is_valid,
    SCORE_THRESHOLD, LOST_N_FRAMES, RECOVER_N_FRAMES, MAX_BBOX_AREA_FRAC, MAX_BBOX_ASPECT_RATIO,
)
from stage4_gmc import (
    build_flow_feature_mask, estimate_motion, GMC_DOWNSCALE,
    GMC_MAX_CORNERS, GMC_QUALITY_LEVEL, GMC_MIN_DISTANCE, GMC_MIN_MATCHES, GMC_RANSAC_THRESH,
)

BOX_SIZE = 40
WINDOW_NAME = "Stage 5.5.1 - Reliability-Weighted Fusion (Stage 2: guarded acting + unified output)"
DEFAULT_MODEL = "models/object_tracking_vittrack_2023sep.onnx"

INIT_REFINE_SEARCH_RADIUS = 8   # px; how far to search around the click for a better anchor center
INIT_REFINE_STEP = 2            # px; grid step size for that search

ORB_NFEATURES = 500
ORB_EDGE_THRESHOLD = 8    # 31 default leaves no valid area on a small template
ORB_PATCH_SIZE = 16
MIN_ORB_CROP_SIZE = ORB_PATCH_SIZE * 2
ORB_RATIO_TEST = 0.75
ORB_MIN_MATCHES = 10
ORB_TEMPLATE_PADDING = 25
TEMPLATE_REFRESH_FRAMES = 30

CANDIDATE_CLUSTER_DISTANCE = 60
CANDIDATE_MIN_RATIO = 0.5

OCCLUSION_MARGIN = 10
LINE_GRACE_COOLDOWN = 10

CONFIRM_FRAMES = 20
RECOVERY_SCORE_THRESHOLD = 0.42  # distinctly above SCORE_THRESHOLD (0.30)
PROBATION_COUNT_DECAY = 2  # a sub-recovery-score frame costs this much progress instead of all of it
TENTATIVE_TRUST_MIN_FRAMES = 75  # a tentative that tracked this long anchors the resumed search
                                  # at its own last position instead of the stale confirmed one

STATIC_DIFF_THRESHOLD = 2.0
STATIC_GUARD_MIN_EXPECTED_MOTION = 1.5  # px/frame; below this the scene is expected to look
                                         # static here (e.g. zoom focus of expansion), so the
                                         # static-region test cannot separate object from HUD

NEAR_CROSSHAIR_RADIUS = 45
HOLD_CORNER_NEIGHBORHOOD = 70
HOLD_MAX_CORNERS = 25
MAX_HOLD_FRAMES = 90

POS_HISTORY_LEN = 5
SEARCH_BASE_RADIUS = 100
SEARCH_RADIUS_GROWTH = 6
SEARCH_MAX_RADIUS = 300
MAX_ACCEPT_DISTANCE = 320
MAX_LOST_FRAMES = 200
ABANDON_RETRY_INTERVAL = 150

MOTION_FALLBACK_MIN_LOST_FRAMES = 60
MOTION_FALLBACK_INTERVAL = 30

SUPPORTER_SEARCH_RADIUS = 250
SUPPORTER_COUNT = 5
SUPPORTER_TARGET_EXCLUSION_MARGIN = 10
GOOD_FEATURES_MAX_CANDIDATES = 60
GOOD_FEATURES_QUALITY_LEVEL = 0.01
GOOD_FEATURES_MIN_DISTANCE = 10
CORNER_EIGEN_BLOCK_SIZE = 5
CORNER_EIGEN_KSIZE = 3
CORNER_LINE_RATIO_MIN = 0.15  # min/max eigenvalue floor; below = edge/line, not corner
SUPPORTER_MIN_BEARING_SEP_DEG = 40
SUPPORTER_MIN_BEARING_SEP_FLOOR_DEG = 15
SUPPORTER_MIN_STRENGTH_RATIO = 0.5

CONTRAST_SMALL_KSIZE = 5
CONTRAST_LARGE_KSIZE = 25
CONTRAST_QUALITY_LEVEL = 0.2
CONTRAST_MIN_DISTANCE = 10
COMPACTNESS_WINDOW_SIZE = 60
COMPACTNESS_MAX_ASPECT = 2.5
CANDIDATE_MERGE_DISTANCE = 12

STATIC_HUD_DIFF_THRESHOLD = 4

SUPPORTER_KLT_WINDOW = 21
SUPPORTER_KLT_MAX_LEVEL = 3
SUPPORTER_GMC_MAX_DEVIATION = 15
SUPPORTER_TRACK_WEIGHT_DECAY = 0.34
SUPPORTER_TRACK_WEIGHT_RECOVERY = 0.05
SUPPORTER_TRACK_MIN_WEIGHT = 0.3

SUPPORTER_VOTE_TOLERANCE = 40
SUPPORTER_MIN_ACTIVE_FOR_CONSENSUS = 2
SUPPORTER_LINE_EXCLUSION_MARGIN = 8

SUPPORTER_REFRESH_INTERVAL = 80
SUPPORTER_REFRESH_MIN_ACTIVE = 4

SUSPICIOUS_SCORE_THRESHOLD = 0.5
SUSPICIOUS_CONSENSUS_DIST = 60

ORB_RELIABILITY_MATCH_SATURATION = 50
ORB_RELIABILITY_SPREAD_SCALE = 100
CONSENSUS_RELIABILITY_VOTE_SATURATION = SUPPORTER_COUNT
CONSENSUS_RELIABILITY_SPREAD_SCALE = SUPPORTER_VOTE_TOLERANCE
GMC_RELIABILITY_DECAY_HALF_LIFE = 60

FUSION_AGREEMENT_MAX_DIST = 40
MIN_SIGNAL_RELIABILITY = 0.05

WINNER_SWITCH_MARGIN = 0.15
WINNER_SWITCH_PERSISTENCE_FRAMES = 3
STICKY_DECLINE_STREAK_THRESHOLD = 2  # consecutive frames of the sticky winner's OWN reliability
                                      # dropping before its stickiness is waived entirely

SIZE_SCALE_REFERENCE = BOX_SIZE
SIZE_SCALE_MIN = 1.0
SIZE_SCALE_MAX = 4.0
SIZE_SCALE_EMA_ALPHA = 0.1

SCALE_RELATIVE_GATES = True  # validity caps grow with the confirmed object's own measured size
GATE_AREA_GROWTH_FACTOR = 4.0
GATE_ASPECT_RELAX = 1.5

TEMPLATE_REFRESH_AGREEMENT_MAX_DIST = 10
TEMPLATE_REFRESH_MIN_RELIABILITY = 0.6
TEMPLATE_REFRESH_MIN_COOLDOWN = 20
CONSENSUS_GATED_REFRESH_ENABLED = False  # disabled: destabilized re-detection in testing


def build_static_hud_mask(gray1, gray2, threshold=STATIC_HUD_DIFF_THRESHOLD):
    if gray2 is None:
        return None
    diff = cv2.absdiff(gray1, gray2)
    return (diff < threshold).astype(np.uint8) * 255


def make_orb():
    return cv2.ORB_create(nfeatures=ORB_NFEATURES, edgeThreshold=ORB_EDGE_THRESHOLD,
                           patchSize=ORB_PATCH_SIZE)


def crop_orb_template(frame, bbox, padding=ORB_TEMPLATE_PADDING):
    x, y, w, h = [int(round(v)) for v in bbox]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(frame.shape[1], x + w + padding)
    y1 = min(frame.shape[0], y + h + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def refine_init_center(frame, point, orb_detector, box_size,
                        search_radius=INIT_REFINE_SEARCH_RADIUS, step=INIT_REFINE_STEP):
    """Search a small grid around `point` for the box center whose padded
    ORB template has the most keypoints, and return that center instead.
    Falls back to `point` unchanged if nothing scores higher."""
    px, py = point
    best_point, best_count = point, -1
    for dx in range(-search_radius, search_radius + 1, step):
        for dy in range(-search_radius, search_radius + 1, step):
            cx, cy = px + dx, py + dy
            box = (cx - box_size // 2, cy - box_size // 2, box_size, box_size)
            template = crop_orb_template(frame, box)
            if template is None:
                continue
            n_kp = len(orb_detector.detect(template, None))
            if n_kp > best_count:
                best_count = n_kp
                best_point = (cx, cy)
    return best_point, best_count


def estimate_motion_with_inliers(prev_gray, curr_gray, feature_mask_small, scale=GMC_DOWNSCALE):
    """Returns (M, n_matches, inlier_ratio); M/inlier_ratio are None/0.0 if too few matches."""
    small_prev = cv2.resize(prev_gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(curr_gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)

    pts_prev = cv2.goodFeaturesToTrack(small_prev, maxCorners=GMC_MAX_CORNERS,
                                        qualityLevel=GMC_QUALITY_LEVEL,
                                        minDistance=max(1, GMC_MIN_DISTANCE // scale),
                                        mask=feature_mask_small)
    if pts_prev is None or len(pts_prev) < GMC_MIN_MATCHES:
        return None, (0 if pts_prev is None else len(pts_prev)), 0.0

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(small_prev, small_curr, pts_prev, None)
    status = status.reshape(-1).astype(bool)
    good_prev = pts_prev[status] * scale
    good_curr = pts_curr[status] * scale
    if len(good_prev) < GMC_MIN_MATCHES:
        return None, len(good_prev), 0.0

    M, inlier_mask = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC,
                                                  ransacReprojThreshold=GMC_RANSAC_THRESH)
    n_matches = len(good_prev)
    if M is None or inlier_mask is None:
        return None, n_matches, 0.0
    return M, n_matches, float(inlier_mask.sum()) / n_matches


def clip01(x):
    return max(0.0, min(1.0, x))


def vit_reliability(score):
    return clip01(score)


def orb_reliability(n_matches, spread):
    if spread is None:
        return 0.0
    match_score = clip01(n_matches / ORB_RELIABILITY_MATCH_SATURATION)
    spread_score = clip01(1.0 - spread / ORB_RELIABILITY_SPREAD_SCALE)
    return min(match_score, spread_score)


def consensus_reliability(n_votes, spread):
    if spread is None:
        return 0.0
    vote_score = clip01(n_votes / CONSENSUS_RELIABILITY_VOTE_SATURATION)
    spread_score = clip01(1.0 - spread / CONSENSUS_RELIABILITY_SPREAD_SCALE)
    return min(vote_score, spread_score)


def gmc_reliability(inlier_ratio, frames_since_anchor):
    decay = 0.5 ** (frames_since_anchor / GMC_RELIABILITY_DECAY_HALF_LIFE)
    return clip01(inlier_ratio) * decay


def compute_shadow_fusion(signals):
    """Logging-only blend; returns (fused_pos, contributors) or (None, [])."""
    usable = [(name, pos, rel) for name, pos, rel in signals if rel is not None and rel >= MIN_SIGNAL_RELIABILITY]
    if not usable:
        return None, []
    total_rel = sum(rel for _, _, rel in usable)
    fx = sum(rel * pos[0] for _, pos, rel in usable) / total_rel
    fy = sum(rel * pos[1] for _, pos, rel in usable) / total_rel
    contributors = [(name, rel / total_rel) for name, _, rel in usable]
    return (fx, fy), contributors


def size_toughness_scale(confirmed_size, reference=SIZE_SCALE_REFERENCE,
                          min_scale=SIZE_SCALE_MIN, max_scale=SIZE_SCALE_MAX):
    if confirmed_size is None or confirmed_size <= 0:
        return 1.0
    return max(min_scale, min(max_scale, confirmed_size / reference))


def compute_guarded_final_position(signals, sticky_winner, challenger_name, challenger_streak,
                                    sticky_prev_rel, sticky_decline_streak,
                                    max_agreement_dist=FUSION_AGREEMENT_MAX_DIST,
                                    switch_margin=WINNER_SWITCH_MARGIN,
                                    switch_persistence=WINNER_SWITCH_PERSISTENCE_FRAMES,
                                    decline_streak_threshold=STICKY_DECLINE_STREAK_THRESHOLD):
    """Returns (final_pos, mode, detail, new_sticky_winner, new_challenger_name,
    new_challenger_streak, new_sticky_prev_rel, new_sticky_decline_streak);
    mode is "none"/"single"/"blended"/"winner-take-all". `sticky_prev_rel`/
    `sticky_decline_streak` track whether the CURRENT sticky winner's own
    reliability has been dropping for consecutive frames -- once it has for
    decline_streak_threshold frames, its stickiness is waived entirely (a
    winner that's actively getting worse shouldn't be protected from a
    switch just because no challenger has "clearly" overtaken it yet)."""
    usable = [(name, pos, rel) for name, pos, rel in signals if rel is not None and rel >= MIN_SIGNAL_RELIABILITY]
    if not usable:
        return None, "none", "", None, None, 0, None, 0
    if len(usable) == 1:
        name, pos, rel = usable[0]
        return pos, "single", f"only {name} available (rel={rel:.2f})", name, None, 0, rel, 0

    positions = [pos for _, pos, _ in usable]
    max_dist = max(math.hypot(a[0] - b[0], a[1] - b[1])
                    for i, a in enumerate(positions) for b in positions[i + 1:])
    if max_dist <= max_agreement_dist:
        total_rel = sum(rel for _, _, rel in usable)
        fx = sum(rel * pos[0] for _, pos, rel in usable) / total_rel
        fy = sum(rel * pos[1] for _, pos, rel in usable) / total_rel
        weights = ", ".join(f"{name}={rel / total_rel:.2f}" for name, _, rel in usable)
        return ((fx, fy), "blended",
                f"agree within {max_dist:.1f}px (<= {max_agreement_dist}) | weights=[{weights}]",
                None, None, 0, None, 0)

    by_name = {name: (pos, rel) for name, pos, rel in usable}
    naive_name, naive_pos, naive_rel = max(usable, key=lambda t: t[2])

    if sticky_winner not in by_name:
        return (naive_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- adopting {naive_name} "
                f"(rel={naive_rel:.2f}), no previous sticky winner active",
                naive_name, None, 0, naive_rel, 0)

    sticky_pos, sticky_rel = by_name[sticky_winner]
    decline_streak = sticky_decline_streak + 1 if (sticky_prev_rel is not None
                                                    and sticky_rel < sticky_prev_rel) else 0
    declining = decline_streak >= decline_streak_threshold

    if naive_name == sticky_winner:
        return (sticky_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- holding {sticky_winner} "
                f"(rel={sticky_rel:.2f}), the naive best again",
                sticky_winner, None, 0, sticky_rel, decline_streak)

    if declining:
        return (naive_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- {sticky_winner} declining "
                f"{decline_streak} frames (rel={sticky_rel:.2f}) -- switching to {naive_name} "
                f"(rel={naive_rel:.2f}) immediately, stickiness waived",
                naive_name, None, 0, naive_rel, 0)

    if naive_rel < sticky_rel + switch_margin:
        return (sticky_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- holding {sticky_winner} "
                f"(rel={sticky_rel:.2f}); {naive_name} (rel={naive_rel:.2f}) ahead but not by the "
                f"{switch_margin} margin required to switch",
                sticky_winner, None, 0, sticky_rel, decline_streak)

    new_streak = challenger_streak + 1 if challenger_name == naive_name else 1
    if new_streak >= switch_persistence:
        return (naive_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- SWITCHED to {naive_name} "
                f"(rel={naive_rel:.2f}), ahead of {sticky_winner} (rel={sticky_rel:.2f}) by >= "
                f"{switch_margin} for {new_streak} frames",
                naive_name, naive_name, new_streak, naive_rel, 0)
    return (sticky_pos, "winner-take-all",
            f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- holding {sticky_winner} "
            f"(rel={sticky_rel:.2f}); {naive_name} ahead by >= {switch_margin} for "
            f"{new_streak}/{switch_persistence} frames, not yet enough to switch",
            sticky_winner, naive_name, new_streak, sticky_rel, decline_streak)


def cluster_orb_matches(pts, cluster_dist):
    clusters = []
    for x, y in pts:
        placed = False
        for c in clusters:
            ccx, ccy = c["centroid"]
            if math.hypot(x - ccx, y - ccy) <= cluster_dist:
                c["pts"].append((x, y))
                n = len(c["pts"])
                c["centroid"] = (ccx + (x - ccx) / n, ccy + (y - ccy) / n)
                placed = True
                break
        if not placed:
            clusters.append({"pts": [(x, y)], "centroid": (x, y)})
    return clusters


def orb_locate_candidates(orb, bf, search_gray, template_entries, box_size, offset=(0, 0),
                           cluster_dist=CANDIDATE_CLUSTER_DISTANCE, min_matches=None,
                           min_ratio=CANDIDATE_MIN_RATIO):
    """Returns list of (bbox, n_matches, spread) in full-frame coords, sorted by n_matches desc."""
    if min_matches is None:
        min_matches = ORB_MIN_MATCHES
    if search_gray.shape[0] < MIN_ORB_CROP_SIZE or search_gray.shape[1] < MIN_ORB_CROP_SIZE:
        return []
    kp2, des2 = orb.detectAndCompute(search_gray, None)
    if des2 is None or len(des2) < 2:
        return []

    best_pts = None
    best_total = 0
    for kp1, des1 in template_entries:
        if des1 is None or len(des1) < 2:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < ORB_RATIO_TEST * n.distance]
        if len(good) < min_matches or len(good) <= best_total:
            continue
        best_total = len(good)
        best_pts = [kp2[m.trainIdx].pt for m in good]

    if best_pts is None:
        return []

    bw, bh = box_size

    def make_bbox(pts):
        arr = np.array(pts)
        cx, cy = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        return (cx - bw / 2 + offset[0], cy - bh / 2 + offset[1], float(bw), float(bh))

    def make_spread(pts):
        arr = np.array(pts)
        median = np.median(arr, axis=0)
        return float(np.mean(np.linalg.norm(arr - median, axis=1)))

    clusters = sorted(
        (c for c in cluster_orb_matches(best_pts, cluster_dist) if len(c["pts"]) >= min_matches),
        key=lambda c: -len(c["pts"]))

    if len(clusters) <= 1 or len(clusters[1]["pts"]) < min_ratio * len(clusters[0]["pts"]):
        return [(make_bbox(best_pts), best_total, make_spread(best_pts))]

    threshold = min_ratio * len(clusters[0]["pts"])
    candidates = [(make_bbox(c["pts"]), len(c["pts"]), make_spread(c["pts"]))
                  for c in clusters if len(c["pts"]) >= threshold]
    candidates.sort(key=lambda t: -t[1])
    return candidates


def expected_gmc_displacement(bbox, M):
    if M is None:
        return None
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    nx, ny = transform_point((cx, cy), M)
    return math.hypot(nx - cx, ny - cy)


def static_guard_applies(bbox, M, min_expected=STATIC_GUARD_MIN_EXPECTED_MOTION):
    disp = expected_gmc_displacement(bbox, M)
    return disp is None or disp >= min_expected


def is_static_region(curr_frame, prev_frame, bbox, threshold=STATIC_DIFF_THRESHOLD):
    x, y, w, h = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(curr_frame.shape[1], x + w), min(curr_frame.shape[0], y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return True
    curr_region = curr_frame[y0:y1, x0:x1]
    prev_region = prev_frame[y0:y1, x0:x1]
    return float(cv2.absdiff(curr_region, prev_region).mean()) < threshold


def is_valid_for_recovery(bbox, score, frame_area, max_area=None, max_aspect=None):
    if score < RECOVERY_SCORE_THRESHOLD:
        return False
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if max_area is None:
        max_area = MAX_BBOX_AREA_FRAC * frame_area
    if max_aspect is None:
        max_aspect = MAX_BBOX_ASPECT_RATIO
    if (w * h) > max_area:
        return False
    return max(w, h) <= max_aspect * min(w, h)


def estimate_velocity(pos_history):
    if len(pos_history) < 3:
        return 0.0, 0.0
    (x0, y0), (x1, y1) = pos_history[-3], pos_history[-1]
    return (x1 - x0) / 2.0, (y1 - y0) / 2.0


def transform_point(pt, M):
    arr = np.array([[pt]], dtype=np.float32)
    warped = cv2.transform(arr, M)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


def transform_vector(v, M):
    """Transforms a vector (not a point) through M's linear part only, dropping translation."""
    linear = np.array(M[:, :2], dtype=np.float64)
    dx, dy = linear @ np.array([v[0], v[1]], dtype=np.float64)
    return float(dx), float(dy)


def find_hold_corners(gray1, point, overlay_mask, neighborhood=HOLD_CORNER_NEIGHBORHOOD,
                       max_corners=HOLD_MAX_CORNERS):
    x0 = max(0, point[0] - neighborhood)
    y0 = max(0, point[1] - neighborhood)
    x1 = min(gray1.shape[1], point[0] + neighborhood)
    y1 = min(gray1.shape[0], point[1] + neighborhood)
    roi_mask = np.zeros(gray1.shape, np.uint8)
    roi_mask[y0:y1, x0:x1] = 255
    roi_mask[overlay_mask > 0] = 0
    return cv2.goodFeaturesToTrack(gray1, maxCorners=max_corners, qualityLevel=0.01,
                                    minDistance=5, mask=roi_mask)


def _bearing_deg(dx, dy):
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _circular_sep_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _find_contrast_peaks(region, region_mask, min_distance=CONTRAST_MIN_DISTANCE,
                          quality_level=CONTRAST_QUALITY_LEVEL,
                          small_ksize=CONTRAST_SMALL_KSIZE, large_ksize=CONTRAST_LARGE_KSIZE):
    """Returns (peaks, contrast_map); peaks is (x, y, response) in region-local coords."""
    small = cv2.GaussianBlur(region, (small_ksize, small_ksize), 0)
    large = cv2.GaussianBlur(region, (large_ksize, large_ksize), 0)
    contrast = cv2.absdiff(small, large)
    contrast[region_mask == 0] = 0

    peak_val = int(contrast.max())
    if peak_val <= 0:
        return [], contrast
    dilated = cv2.dilate(contrast, np.ones((min_distance, min_distance), np.uint8))
    threshold = max(1, int(quality_level * peak_val))
    peak_mask = (contrast == dilated) & (contrast >= threshold)
    ys, xs = np.where(peak_mask)
    return list(zip(xs.tolist(), ys.tolist(), contrast[ys, xs].tolist())), contrast


def _is_compact_blob(contrast_map, x, y, peak_value, window=COMPACTNESS_WINDOW_SIZE,
                      max_aspect=COMPACTNESS_MAX_ASPECT):
    """Returns (is_compact, aspect)."""
    half = window // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(contrast_map.shape[1], x + half), min(contrast_map.shape[0], y + half)
    local = contrast_map[y0:y1, x0:x1]
    thresh = max(1, int(0.5 * peak_value))
    binary = (local >= thresh).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lx, ly = x - x0, y - y0
    if not (0 <= ly < labels.shape[0] and 0 <= lx < labels.shape[1]):
        return False, float("inf")
    label = labels[ly, lx]
    if label == 0:
        return False, float("inf")

    bx, by, bw, bh, _ = stats[label]
    aspect = max(bw, bh) / max(1, min(bw, bh))
    touches_border = bx <= 0 or by <= 0 or bx + bw >= local.shape[1] or by + bh >= local.shape[0]
    return (aspect <= max_aspect and not touches_border), float(aspect)


def select_supporters(gray, target_bbox, overlay_mask, static_mask=None, search_radius=SUPPORTER_SEARCH_RADIUS,
                       target_count=SUPPORTER_COUNT, min_bearing_sep=SUPPORTER_MIN_BEARING_SEP_DEG,
                       corner_line_ratio_min=CORNER_LINE_RATIO_MIN,
                       min_strength_ratio=SUPPORTER_MIN_STRENGTH_RATIO):
    """Returns (chosen, n_raw, n_survivors); chosen is a list of dicts with
    keys pt/offset/strength/source/ratio/bearing."""
    tx, ty, tw, th = target_bbox
    tcx, tcy = tx + tw / 2, ty + th / 2

    x0 = int(max(VX0, tcx - search_radius))
    y0 = int(max(0, tcy - search_radius))
    x1 = int(min(gray.shape[1], VX1, tcx + search_radius))
    y1 = int(min(gray.shape[0], tcy + search_radius))
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return [], 0, 0

    region_mask = np.full(region.shape, 255, np.uint8)
    region_mask[overlay_mask[y0:y1, x0:x1] > 0] = 0
    if static_mask is not None:
        region_mask[static_mask[y0:y1, x0:x1] > 0] = 0
    m = SUPPORTER_TARGET_EXCLUSION_MARGIN
    ex0 = int(max(0, tx - m - x0))
    ey0 = int(max(0, ty - m - y0))
    ex1 = int(min(region.shape[1], tx + tw + m - x0))
    ey1 = int(min(region.shape[0], ty + th + m - y0))
    region_mask[ey0:ey1, ex0:ex1] = 0

    corner_raw = cv2.goodFeaturesToTrack(region, maxCorners=GOOD_FEATURES_MAX_CANDIDATES,
                                         qualityLevel=GOOD_FEATURES_QUALITY_LEVEL,
                                         minDistance=GOOD_FEATURES_MIN_DISTANCE, mask=region_mask)
    corner_pts = [] if corner_raw is None else [(p[0], p[1]) for p in corner_raw.reshape(-1, 2)]
    contrast_peaks, contrast_map = _find_contrast_peaks(region, region_mask)
    n_raw = len(corner_pts) + len(contrast_peaks)
    if n_raw == 0:
        return [], 0, 0

    eigen = cv2.cornerEigenValsAndVecs(region, blockSize=CORNER_EIGEN_BLOCK_SIZE, ksize=CORNER_EIGEN_KSIZE)

    def eigen_ratio(px, py):
        if not (0 <= px < region.shape[1] and 0 <= py < region.shape[0]):
            return None
        e1, e2 = eigen[py, px, 0], eigen[py, px, 1]
        min_eig, max_eig = (e1, e2) if e1 <= e2 else (e2, e1)
        if max_eig <= 1e-12:
            return None
        return float(min_eig), float(min_eig / max_eig)

    survivors_by_source = {"corner": [], "contrast": []}
    for rx, ry in corner_pts:
        px, py = int(round(rx)), int(round(ry))
        r = eigen_ratio(px, py)
        if r is None or r[1] < corner_line_ratio_min:
            continue
        survivors_by_source["corner"].append((rx, ry, r[0], r[1]))
    for px, py, peak_value in contrast_peaks:
        is_compact, aspect = _is_compact_blob(contrast_map, px, py, peak_value)
        if not is_compact:
            continue
        survivors_by_source["contrast"].append((px, py, float(peak_value), 1.0 / aspect))

    n_survivors = len(survivors_by_source["corner"]) + len(survivors_by_source["contrast"])

    merged = []
    for source, entries in survivors_by_source.items():
        if not entries:
            continue
        max_strength = max(e[2] for e in entries)
        for rx, ry, strength, ratio in entries:
            gx, gy = rx + x0, ry + y0
            merged.append({
                "pt": (float(gx), float(gy)),
                "offset": (float(gx - tcx), float(gy - tcy)),
                "strength": strength / max_strength,
                "source": source,
                "ratio": ratio,
                "bearing": _bearing_deg(gx - tcx, gy - tcy),
                "weight": 1.0,
                "active": True,
            })

    merged.sort(key=lambda s: -s["strength"])
    deduped = []
    for s in merged:
        sx, sy = s["pt"]
        if any(math.hypot(sx - d["pt"][0], sy - d["pt"][1]) <= CANDIDATE_MERGE_DISTANCE for d in deduped):
            continue
        deduped.append(s)

    if not deduped:
        return [], n_raw, n_survivors

    strength_floor = min_strength_ratio * deduped[0]["strength"]
    eligible = [s for s in deduped if s["strength"] >= strength_floor]

    sep_schedule = sorted({min_bearing_sep, min_bearing_sep / 2, SUPPORTER_MIN_BEARING_SEP_FLOOR_DEG},
                          reverse=True)
    chosen = []
    for min_sep in sep_schedule:
        if len(chosen) >= target_count:
            break
        for s in eligible:
            if len(chosen) >= target_count:
                break
            if s in chosen:
                continue
            if all(_circular_sep_deg(s["bearing"], c["bearing"]) >= min_sep for c in chosen):
                chosen.append(s)

    return chosen, n_raw, n_survivors


def update_supporters(supporters, prev_gray, curr_gray, motion_M, width, height, frame_idx):
    """Mutates `supporters` in place."""
    active = [s for s in supporters if s["active"]]
    if not active:
        return

    pts = np.float32([s["pt"] for s in active]).reshape(-1, 1, 2)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize=(SUPPORTER_KLT_WINDOW, SUPPORTER_KLT_WINDOW), maxLevel=SUPPORTER_KLT_MAX_LEVEL)
    status = status.reshape(-1)

    for i, s in enumerate(active):
        if not status[i]:
            s["weight"] = max(0.0, s["weight"] - SUPPORTER_TRACK_WEIGHT_DECAY)
            if s["weight"] <= SUPPORTER_TRACK_MIN_WEIGHT:
                s["active"] = False
                print(f"Frame {frame_idx} | Supporter dropped (optical-flow lost track) | "
                      f"pt=({s['pt'][0]:.1f},{s['pt'][1]:.1f})")
            continue

        nx, ny = float(new_pts[i, 0, 0]), float(new_pts[i, 0, 1])

        if not (VX0 <= nx <= VX1 and 0 <= ny <= height):
            s["active"] = False
            print(f"Frame {frame_idx} | Supporter dropped (left frame bounds) | "
                  f"pt=({nx:.1f},{ny:.1f})")
            continue

        rigid = True
        if motion_M is not None:
            pred_x, pred_y = transform_point(s["pt"], motion_M)
            deviation = math.hypot(nx - pred_x, ny - pred_y)
            rigid = deviation <= SUPPORTER_GMC_MAX_DEVIATION

        s["pt"] = (nx, ny)
        if rigid:
            if motion_M is not None:
                s["offset"] = transform_vector(s["offset"], motion_M)
            s["weight"] = min(1.0, s["weight"] + SUPPORTER_TRACK_WEIGHT_RECOVERY)
        else:
            s["weight"] = max(0.0, s["weight"] - SUPPORTER_TRACK_WEIGHT_DECAY)
            if s["weight"] <= SUPPORTER_TRACK_MIN_WEIGHT:
                s["active"] = False
                print(f"Frame {frame_idx} | Supporter dropped (non-rigid -- GMC motion "
                      f"mismatch {deviation:.1f}px) | pt=({nx:.1f},{ny:.1f})")


def compute_supporter_consensus(supporters, overlay_mask=None, line_margin=SUPPORTER_LINE_EXCLUSION_MARGIN,
                                 tolerance=SUPPORTER_VOTE_TOLERANCE,
                                 min_active=SUPPORTER_MIN_ACTIVE_FOR_CONSENSUS):
    """Returns (consensus_pos, n_votes_used, spread) or (None, 0, None)."""
    active = [s for s in supporters if s["active"]]
    if overlay_mask is not None:
        active = [s for s in active
                   if not bbox_overlaps_mask((s["pt"][0] - 1, s["pt"][1] - 1, 2, 2), overlay_mask, line_margin)]
    if len(active) < min_active:
        return None, 0, None

    votes = np.array([(s["pt"][0] - s["offset"][0], s["pt"][1] - s["offset"][1]) for s in active])
    median_vote = np.median(votes, axis=0)
    deviations = np.linalg.norm(votes - median_vote, axis=1)
    inliers = votes[deviations <= tolerance]

    if len(inliers) < min_active:
        spread = float(np.mean(deviations))
        return (float(median_vote[0]), float(median_vote[1])), len(active), spread

    final = np.median(inliers, axis=0)
    spread = float(np.mean(np.linalg.norm(inliers - final, axis=1)))
    return (float(final[0]), float(final[1])), int(len(inliers)), spread


def init_supporters(gray, init_box, overlay_mask, static_mask=None):
    supporters, n_raw, n_survivors = select_supporters(gray, init_box, overlay_mask, static_mask=static_mask)
    hud_note = "" if static_mask is None else " (screen-locked HUD pixels excluded)"
    print(f"Supporter selection: {n_raw} raw candidates (corner + contrast-blob sources){hud_note} -> "
          f"{n_survivors} passed the corner-vs-line filter (ratio >= {CORNER_LINE_RATIO_MIN}) -> "
          f"{len(supporters)} chosen (target {SUPPORTER_COUNT}, min strength ratio "
          f"{SUPPORTER_MIN_STRENGTH_RATIO}, min bearing separation {SUPPORTER_MIN_BEARING_SEP_DEG}-"
          f"{SUPPORTER_MIN_BEARING_SEP_FLOOR_DEG} deg).")
    for i, s in enumerate(supporters):
        print(f"  supporter #{i} [{s['source']}]: pt=({s['pt'][0]:.1f},{s['pt'][1]:.1f}) "
              f"offset=({s['offset'][0]:.1f},{s['offset'][1]:.1f}) bearing={s['bearing']:.1f}deg "
              f"strength={s['strength']:.2f} ratio={s['ratio']:.2f}")
    if not supporters:
        print("  No qualifying supporters found in this neighborhood (few strong, compact, "
              "well-separated features) -- context-aware disambiguation will be unavailable for this target.")
    return supporters


def estimate_held_point(gray1, curr_gray, corners0, click_point):
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


def draw_final_marker(frame, final_pos, box_wh, color=(0, 255, 0)):
    cx, cy = final_pos
    w, h = int(round(box_wh[0])), int(round(box_wh[1]))
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
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
    flow_mask = build_flow_feature_mask(width, height, cleaner.full_mask)
    flow_mask_small = cv2.resize(flow_mask, None, fx=1 / GMC_DOWNSCALE, fy=1 / GMC_DOWNSCALE,
                                  interpolation=cv2.INTER_NEAREST)

    display_scale = compute_display_scale(width, height)
    if display_scale < 1.0:
        print(f"Display scaled to {display_scale:.2f}x to fit the screen "
              f"({width}x{height} -> {int(width * display_scale)}x{int(height * display_scale)})")

    ok, frame1 = cap.read()
    if not ok:
        print("ERROR: could not read first frame.")
        sys.exit(1)

    ok2, pending_frame = cap.read()
    peeked_gray = cv2.cvtColor(pending_frame, cv2.COLOR_BGR2GRAY) if ok2 else None
    if not ok2:
        pending_frame = None
    static_mask1 = build_static_hud_mask(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), peeked_gray)

    if args.x is not None and args.y is not None:
        point = (args.x, args.y)
        print(f"Using manual point: {point}")
    else:
        point = pick_point_by_click(frame1, display_scale=display_scale, window_name=WINDOW_NAME)
        print(f"Picked point: {point}")

    cleaned1 = cleaner.clean(frame1, hint_center=point)

    half = args.box_size // 2
    params = cv2.TrackerVit_Params()
    params.net = args.model
    orb = make_orb()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    refined_point, refined_kp = refine_init_center(cleaned1, point, orb, args.box_size)
    if refined_point != point:
        print(f"Refined init point: {point} -> {refined_point} ({refined_kp} keypoints)")
    point = refined_point

    cv2.namedWindow(WINDOW_NAME)
    show_cleaned = args.show_cleaned
    show_mask = args.show_mask

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{video_stem}_x{point[0]}_y{point[1]}_{timestamp}.mp4")
    writer_fps = src_fps if src_fps and src_fps > 0 else 30.0
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (width, height))
    print(f"Recording output to: {output_path}")

    frame_idx = 0
    fps_window_start = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    cleaned = cleaned1
    prev_cleaned = cleaned1
    frame = frame1
    prev_frame_raw = frame1

    bad_count = 0
    probation_frames = 0
    tentative_last_pos = None
    frames_since_refresh = 0
    frames_since_supporter_refresh = 0
    line_grace_remaining = 0
    lost_frames = 0
    frames_since_motion_fallback = 0
    match_count_display = 0
    search_mode_display = "-"
    n_lost_frames = 0
    n_transitions = 0
    n_static_rejections = 0
    n_distance_rejections = 0

    predicted_pos = None
    lost_prev_gray = None
    search_abandoned = False
    search_cycle_frames = 0
    frames_since_abandon = 0
    n_retry_cycles = 0
    n_ambiguous_resolutions = 0
    n_suspicious_disagreements = 0
    n_shadow_fusion_frames = 0
    n_shadow_fusion_multi_signal = 0
    n_winner_take_all_frames = 0
    n_consensus_gated_refreshes = 0
    n_winner_switches = 0
    sticky_winner = None
    challenger_name = None
    challenger_streak = 0
    sticky_prev_rel = None
    sticky_decline_streak = 0
    confirmed_object_size = float(BOX_SIZE)
    confirmed_aspect = 1.0
    last_logged_size_scale = 1.0
    final_pos = None
    consensus_pos = None
    n_votes = 0

    tracker = None
    orig_orb = (None, None)
    recent_orb = (None, None)
    supporters = []

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

        supporters = init_supporters(
            cv2.cvtColor(cleaned1, cv2.COLOR_BGR2GRAY), init_box, cleaner.full_mask,
            static_mask=static_mask1)

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
        print(f"Click is {dist_to_crosshair:.0f}px from the crosshair (<= {args.near_crosshair_radius:.0f}) "
              f"-- holding until it clears the crosshair before initializing.")
        state = "HOLDING"
        held_point = (float(point[0]), float(point[1]))
        hold_frames = 0
        gray1_hold = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hold_prev_gray = gray1_hold
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

        if final_pos is not None:
            box_wh = ((bbox[2], bbox[3]) if (bbox_valid_shape and state == "TRACKING")
                      else (args.box_size, args.box_size))
            draw_final_marker(display, final_pos, box_wh)

        if state not in ("TRACKING", "HOLDING"):
            n_lost_frames += 1

        stream_label = "CLEANED" if show_cleaned else "ORIGINAL"
        mask_label = " + MASK" if show_mask else ""
        cv2.putText(display, f"frame {frame_idx}  fps {display_fps:.1f}  [{stream_label}{mask_label}]",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        shown = (cv2.resize(display, (int(width * display_scale), int(height * display_scale)))
                 if display_scale < 1.0 else display)
        cv2.imshow(WINDOW_NAME, shown)
        writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            show_cleaned = not show_cleaned
        elif key == ord("m"):
            show_mask = not show_mask

        if pending_frame is not None:
            frame, ok = pending_frame, True
            pending_frame = None
        else:
            ok, frame = cap.read()
        if not ok:
            print("End of video.")
            break
        frame_idx += 1

        if state == "TRACKING" and bbox is not None:
            clean_hint = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        elif state == "LOST" and predicted_pos is not None:
            clean_hint = predicted_pos
        elif state == "HOLDING":
            clean_hint = held_point
        else:
            clean_hint = None

        prev_cleaned = cleaned
        cleaned = cleaner.clean(frame, hint_center=clean_hint)

        if supporters:
            prev_frame_gray = cv2.cvtColor(prev_frame_raw, cv2.COLOR_BGR2GRAY)
            curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            supporter_motion_M, _ = estimate_motion(prev_frame_gray, curr_frame_gray, flow_mask_small)
            update_supporters(supporters, prev_frame_gray, curr_frame_gray, supporter_motion_M,
                               width, height, frame_idx)
        prev_frame_raw = frame

        gmc_rel = None
        lost_motion_M = None
        uncertain = (state == "LOST") or (state == "TRACKING" and probation)
        if uncertain and lost_prev_gray is not None and predicted_pos is not None:
            curr_gray_gmc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_M, gmc_n_matches, gmc_inlier_ratio = estimate_motion_with_inliers(
                lost_prev_gray, curr_gray_gmc, flow_mask_small)
            lost_motion_M = motion_M
            gmc_rel = gmc_reliability(gmc_inlier_ratio, lost_frames)
            print(f"Frame {frame_idx} | Reliability | gmc={gmc_rel:.2f} "
                  f"(inlier_ratio={gmc_inlier_ratio:.2f}, matches={gmc_n_matches}, "
                  f"frames_since_anchor={lost_frames})")
            if motion_M is not None:
                predicted_pos = transform_point(predicted_pos, motion_M)
            else:
                vx, vy = estimate_velocity(pos_history)
                predicted_pos = (predicted_pos[0] + vx, predicted_pos[1] + vy)
            lost_prev_gray = curr_gray_gmc

        if state == "HOLDING":
            curr_gray_hold = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            estimated = estimate_held_point(gray1_hold, curr_gray_hold, hold_corners0, point)
            if estimated is not None:
                held_point = estimated
            else:
                hold_M, _ = estimate_motion(hold_prev_gray, curr_gray_hold, flow_mask_small)
                if hold_M is not None:
                    held_point = transform_point(held_point, hold_M)
                    print(f"Frame {frame_idx} | HOLDING local flow unavailable -- held point "
                          f"carried by GMC to ({held_point[0]:.1f},{held_point[1]:.1f})")
            hold_prev_gray = curr_gray_hold
            hold_frames += 1

            dist_to_crosshair = math.hypot(held_point[0] - CX, held_point[1] - CY)
            if dist_to_crosshair > args.near_crosshair_radius or hold_frames >= MAX_HOLD_FRAMES:
                hx = min(max(held_point[0], VX0 + half), VX1 - half)
                hy = min(max(held_point[1], half), height - half)
                (hx, hy), refined_kp = refine_init_center(cleaned, (int(round(hx)), int(round(hy))),
                                                           orb, args.box_size)
                init_box = (int(round(hx - half)), int(round(hy - half)), args.box_size, args.box_size)

                tracker = cv2.TrackerVit_create(params)
                tracker.init(cleaned, init_box)
                orig_tmpl_gray = crop_orb_template(cleaned, init_box)
                orig_orb = orb.detectAndCompute(orig_tmpl_gray, None) if orig_tmpl_gray is not None else (None, None)
                recent_orb = orig_orb

                ok2, next_frame = cap.read()
                peeked_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) if ok2 else None
                static_mask_now = build_static_hud_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), peeked_gray)
                pending_frame = next_frame if ok2 else None
                supporters = init_supporters(
                    cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), init_box, cleaner.full_mask,
                    static_mask=static_mask_now)
                frames_since_supporter_refresh = 0

                state = "TRACKING"
                probation = False
                probation_count = 0
                line_grace_remaining = 0
                pos_history = [(hx, hy)]
                last_good_bbox = init_box
                bbox = init_box
                score = 1.0
                reason = "cleared crosshair" if dist_to_crosshair > args.near_crosshair_radius else "max hold reached"
                print(f"Frame {frame_idx} | HOLDING->TRACKING ({reason}) | "
                      f"init_box={init_box} after {hold_frames} held frames, ORB kp={len(orig_orb[0]) if orig_orb[0] else 0}")

        elif state == "TRACKING":
            if SCALE_RELATIVE_GATES:
                dyn_max_area = max(MAX_BBOX_AREA_FRAC * frame_area,
                                   GATE_AREA_GROWTH_FACTOR * confirmed_object_size ** 2)
                dyn_max_aspect = max(MAX_BBOX_ASPECT_RATIO, confirmed_aspect * GATE_ASPECT_RELAX)
            else:
                dyn_max_area = dyn_max_aspect = None
            _, bbox = tracker.update(cleaned)
            score = tracker.getTrackingScore()
            valid = is_valid(bbox, score, frame_area, dyn_max_area, dyn_max_aspect)
            eff_max_area = dyn_max_area if dyn_max_area is not None else MAX_BBOX_AREA_FRAC * frame_area
            eff_max_aspect = dyn_max_aspect if dyn_max_aspect is not None else MAX_BBOX_ASPECT_RATIO
            geo_sane = (bbox[2] > 0 and bbox[3] > 0
                        and (bbox[2] * bbox[3]) <= eff_max_area
                        and max(bbox[2], bbox[3]) <= eff_max_aspect * min(bbox[2], bbox[3]))
            currently_overlapping = geo_sane and bbox_overlaps_mask(bbox, cleaner.full_mask, OCCLUSION_MARGIN)
            if currently_overlapping:
                line_grace_remaining = LINE_GRACE_COOLDOWN
            elif line_grace_remaining > 0:
                line_grace_remaining -= 1
            near_line = geo_sane and (currently_overlapping or line_grace_remaining > 0)

            if probation:
                lost_frames += 1
                frames_since_motion_fallback += 1
                probation_frames += 1
                if valid:
                    tentative_last_pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                    bad_count = 0
                    if is_valid_for_recovery(bbox, score, frame_area, dyn_max_area, dyn_max_aspect):
                        probation_count += 1
                    else:
                        probation_count = max(0, probation_count - PROBATION_COUNT_DECAY)
                    if probation_count >= CONFIRM_FRAMES:
                        probation = False
                        last_good_bbox = bbox
                        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                        pos_history = [(cx, cy)]
                        refreshed_gray = crop_orb_template(cleaned, bbox)
                        if refreshed_gray is not None:
                            recent_orb = orb.detectAndCompute(refreshed_gray, None)
                        frames_since_refresh = 0
                        print(f"Frame {frame_idx} | TENTATIVE->TRACKING (confirmed) | bbox={bbox} | score={score:.3f}")
                else:
                    bad_count += 1
                    probation_count = 0
                    if bad_count >= args.lost_n:
                        state = "LOST"
                        probation = False
                        n_transitions += 1
                        if probation_frames >= TENTATIVE_TRUST_MIN_FRAMES and tentative_last_pos is not None:
                            predicted_pos = tentative_last_pos
                            print(f"Frame {frame_idx} | TENTATIVE->LOST (candidate did not hold) | "
                                  f"score={score:.3f} | resuming search from the {probation_frames}-frame "
                                  f"tentative's last position ({predicted_pos[0]:.1f},{predicted_pos[1]:.1f})")
                        else:
                            print(f"Frame {frame_idx} | TENTATIVE->LOST (candidate did not hold) | "
                                  f"score={score:.3f} | resuming search from last CONFIRMED position")
            elif valid or near_line:
                bad_count = 0
                last_good_bbox = bbox
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                pos_history.append((cx, cy))
                if len(pos_history) > POS_HISTORY_LEN:
                    pos_history.pop(0)
                if valid:
                    frames_since_refresh += 1
                    if frames_since_refresh >= TEMPLATE_REFRESH_FRAMES:
                        refreshed_gray = crop_orb_template(cleaned, bbox)
                        if refreshed_gray is not None:
                            recent_orb = orb.detectAndCompute(refreshed_gray, None)
                        frames_since_refresh = 0

                    frames_since_supporter_refresh += 1
                    n_active_supp = sum(1 for s in supporters if s["active"])
                    if (frames_since_supporter_refresh >= SUPPORTER_REFRESH_INTERVAL
                            or n_active_supp < SUPPORTER_REFRESH_MIN_ACTIVE):
                        reason = ("interval elapsed" if frames_since_supporter_refresh >= SUPPORTER_REFRESH_INTERVAL
                                  else f"only {n_active_supp} active")
                        ok_peek, next_frame = cap.read()
                        peeked_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) if ok_peek else None
                        static_mask_refresh = build_static_hud_mask(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), peeked_gray)
                        pending_frame = next_frame if ok_peek else None

                        supporters, n_raw_r, n_survivors_r = select_supporters(
                            cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), bbox, cleaner.full_mask,
                            static_mask=static_mask_refresh)
                        frames_since_supporter_refresh = 0
                        print(f"Frame {frame_idx} | Supporters refreshed ({reason}) | "
                              f"{n_raw_r} raw -> {n_survivors_r} survivors -> {len(supporters)} chosen")
            else:
                bad_count += 1
                if bad_count >= args.lost_n:
                    state = "LOST"
                    lost_frames = 0
                    search_abandoned = False
                    search_cycle_frames = 0
                    frames_since_abandon = 0
                    frames_since_motion_fallback = 0
                    lcx, lcy = last_good_bbox[0] + last_good_bbox[2] / 2, last_good_bbox[1] + last_good_bbox[3] / 2
                    predicted_pos = (lcx, lcy)
                    lost_prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    n_transitions += 1
                    print(f"Frame {frame_idx} | TRACKING->LOST | score={score:.3f} | bbox={bbox}")
        else:  # LOST -- motion-gated ORB search, tracker.update() not consulted
            lost_frames += 1
            frames_since_motion_fallback += 1
            if search_abandoned:
                frames_since_abandon += 1
                if frames_since_abandon >= ABANDON_RETRY_INTERVAL:
                    search_abandoned = False
                    search_cycle_frames = 0
                    frames_since_abandon = 0
                    n_retry_cycles += 1
                    print(f"Frame {frame_idx} | LOST search resuming (periodic retry after "
                          f"{ABANDON_RETRY_INTERVAL} abandoned frames)")
            else:
                search_cycle_frames += 1
                if search_cycle_frames > MAX_LOST_FRAMES:
                    search_abandoned = True
                    frames_since_abandon = 0
                    print(f"Frame {frame_idx} | LOST search abandoned after {MAX_LOST_FRAMES} frames (TTL) -- "
                          f"will retry again in {ABANDON_RETRY_INTERVAL} frames")

            match_count_display = 0
            if search_abandoned or predicted_pos is None:
                search_mode_display = "ABANDONED" if search_abandoned else "-"
            else:
                pcx, pcy = predicted_pos
                if not (VX0 <= pcx <= VX1 and 0 <= pcy <= height):
                    search_mode_display = "OFF-FRAME"
                else:
                    search_mode_display = "LOCAL"
                    radius = min(SEARCH_BASE_RADIUS + lost_frames * SEARCH_RADIUS_GROWTH, SEARCH_MAX_RADIUS)
                    rx0 = int(max(VX0, pcx - radius))
                    ry0 = int(max(0, pcy - radius))
                    rx1 = int(min(VX1, pcx + radius))
                    ry1 = int(min(height, pcy + radius))
                    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
                    search_region = gray[ry0:ry1, rx0:rx1]

                    candidates = []
                    if search_region.shape[0] >= MIN_ORB_CROP_SIZE and search_region.shape[1] >= MIN_ORB_CROP_SIZE:
                        candidates = orb_locate_candidates(
                            orb, bf, search_region, [orig_orb, recent_orb],
                            box_size=(args.box_size, args.box_size), offset=(rx0, ry0))

                    candidate_bbox, n_matches, candidate_spread = None, 0, None
                    if len(candidates) == 1:
                        candidate_bbox, n_matches, candidate_spread = candidates[0]
                    elif len(candidates) > 1:
                        consensus_pos, n_votes, _ = compute_supporter_consensus(supporters, overlay_mask=cleaner.full_mask)
                        if consensus_pos is not None:
                            scored = []
                            for cbbox, ccount, cspread in candidates:
                                cx2 = cbbox[0] + cbbox[2] / 2
                                cy2 = cbbox[1] + cbbox[3] / 2
                                cdist = math.hypot(cx2 - consensus_pos[0], cy2 - consensus_pos[1])
                                scored.append((cdist, cbbox, ccount, cspread))
                            scored.sort(key=lambda t: t[0])
                            chosen_dist, candidate_bbox, n_matches, candidate_spread = scored[0]
                            others = [(tuple(round(v, 1) for v in b[:2]), c, round(d, 1))
                                      for d, b, c, _ in scored[1:]]
                            n_ambiguous_resolutions += 1
                            print(f"Frame {frame_idx} | LOST (ambiguous: {len(candidates)} candidates, "
                                  f"consensus n_votes={n_votes}) | picked "
                                  f"bbox={tuple(round(v, 1) for v in candidate_bbox)} matches={n_matches} "
                                  f"consensus_dist={chosen_dist:.1f} | others={others}")
                        else:
                            candidate_bbox, n_matches, candidate_spread = candidates[0]
                            print(f"Frame {frame_idx} | LOST (ambiguous: {len(candidates)} candidates, "
                                  f"context unavailable -- no active supporters) | falling back to "
                                  f"best-match bbox={tuple(round(v, 1) for v in candidate_bbox)} "
                                  f"matches={n_matches}")
                    match_count_display = n_matches

                    if candidate_bbox is not None:
                        orb_rel = orb_reliability(n_matches, candidate_spread)
                        print(f"Frame {frame_idx} | Reliability | orb={orb_rel:.2f} "
                              f"(matches={n_matches}, spread={candidate_spread:.1f}px)")

                        ccx = candidate_bbox[0] + candidate_bbox[2] / 2
                        ccy = candidate_bbox[1] + candidate_bbox[3] / 2
                        dist_from_pred = math.hypot(ccx - pcx, ccy - pcy)

                        if dist_from_pred > MAX_ACCEPT_DISTANCE:
                            n_distance_rejections += 1
                            print(f"Frame {frame_idx} | LOST (distance-gate reject) | matches={n_matches} | "
                                  f"predicted=({pcx:.1f},{pcy:.1f}) | candidate=({ccx:.1f},{ccy:.1f}) | "
                                  f"dist={dist_from_pred:.1f} > {MAX_ACCEPT_DISTANCE}")
                        elif (static_guard_applies(candidate_bbox, lost_motion_M)
                              and is_static_region(cleaned, prev_cleaned, candidate_bbox)):
                            n_static_rejections += 1
                            print(f"Frame {frame_idx} | LOST (static-region reject) | matches={n_matches} | "
                                  f"bbox={tuple(round(v, 1) for v in candidate_bbox)}")
                        else:
                            tracker = cv2.TrackerVit_create(params)
                            tracker.init(cleaned, tuple(int(round(v)) for v in candidate_bbox))
                            state = "TRACKING"
                            probation = True
                            probation_count = 0
                            probation_frames = 0
                            tentative_last_pos = None
                            bad_count = 0
                            line_grace_remaining = 0
                            bbox = candidate_bbox
                            n_transitions += 1
                            print(f"Frame {frame_idx} | LOST->TENTATIVE (re-detected, unconfirmed) | matches={n_matches} | "
                                  f"predicted=({pcx:.1f},{pcy:.1f}) | accepted=({ccx:.1f},{ccy:.1f}) | "
                                  f"dist={dist_from_pred:.1f} | radius={radius:.0f}")

            if (state == "LOST" and not search_abandoned and predicted_pos is not None
                    and lost_frames >= MOTION_FALLBACK_MIN_LOST_FRAMES
                    and frames_since_motion_fallback >= MOTION_FALLBACK_INTERVAL):
                frames_since_motion_fallback = 0
                pcx, pcy = predicted_pos
                if VX0 <= pcx <= VX1 and 0 <= pcy <= height:
                    half_b = args.box_size // 2
                    fb_bbox = (int(round(pcx - half_b)), int(round(pcy - half_b)), args.box_size, args.box_size)
                    if (static_guard_applies(fb_bbox, lost_motion_M)
                            and is_static_region(cleaned, prev_cleaned, fb_bbox)):
                        print(f"Frame {frame_idx} | LOST (motion-fallback static-region reject) | bbox={fb_bbox}")
                    else:
                        tracker = cv2.TrackerVit_create(params)
                        tracker.init(cleaned, fb_bbox)
                        state = "TRACKING"
                        probation = True
                        probation_count = 0
                        probation_frames = 0
                        tentative_last_pos = None
                        bad_count = 0
                        line_grace_remaining = 0
                        bbox = fb_bbox
                        n_transitions += 1
                        print(f"Frame {frame_idx} | LOST->TENTATIVE (motion-only fallback) | "
                              f"predicted=({pcx:.1f},{pcy:.1f}) | bbox={fb_bbox}")

        consensus_pos, n_votes, consensus_spread = compute_supporter_consensus(
            supporters, overlay_mask=cleaner.full_mask)
        if state == "TRACKING" and bbox is not None:
            tracker_pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        elif state == "LOST" and predicted_pos is not None:
            tracker_pos = predicted_pos
        else:
            tracker_pos = None
        if consensus_pos is not None and tracker_pos is not None:
            dist = math.hypot(consensus_pos[0] - tracker_pos[0], consensus_pos[1] - tracker_pos[1])
            print(f"Frame {frame_idx} | Consensus check | consensus=({consensus_pos[0]:.1f},"
                  f"{consensus_pos[1]:.1f}) tracker=({tracker_pos[0]:.1f},{tracker_pos[1]:.1f}) "
                  f"dist={dist:.1f} n_votes={n_votes}")
            if (state == "TRACKING" and not probation and score >= SUSPICIOUS_SCORE_THRESHOLD
                    and dist >= SUSPICIOUS_CONSENSUS_DIST):
                n_suspicious_disagreements += 1
                print(f"Frame {frame_idx} | SUSPICIOUS high-confidence disagreement | "
                      f"score={score:.3f} (>= {SUSPICIOUS_SCORE_THRESHOLD}) but consensus disagrees by "
                      f"{dist:.1f}px (>= {SUSPICIOUS_CONSENSUS_DIST}) -- possible confident wrong lock, "
                      f"not yet acted on (monitoring only)")

        bbox_valid_shape = bbox is not None and bbox[2] > 0 and bbox[3] > 0
        vit_pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) if bbox_valid_shape else None
        if state == "TRACKING" and bbox_valid_shape:
            vit_rel = vit_reliability(score)
            print(f"Frame {frame_idx} | Reliability | vit={vit_rel:.2f} (score={score:.3f})")
        else:
            vit_rel = None

        if state == "TRACKING" and not probation and bbox_valid_shape:
            current_size = math.sqrt(bbox[2] * bbox[3])
            confirmed_object_size += SIZE_SCALE_EMA_ALPHA * (current_size - confirmed_object_size)
            current_aspect = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
            confirmed_aspect += SIZE_SCALE_EMA_ALPHA * (current_aspect - confirmed_aspect)
        size_scale = size_toughness_scale(confirmed_object_size)
        if round(size_scale, 2) != round(last_logged_size_scale, 2):
            print(f"Frame {frame_idx} | Size-scaled toughness | confirmed_size={confirmed_object_size:.1f}px "
                  f"(ref={SIZE_SCALE_REFERENCE}) -> scale={size_scale:.2f}x -- "
                  f"agreement_dist={FUSION_AGREEMENT_MAX_DIST * size_scale:.1f}px, "
                  f"switch_persistence={round(WINNER_SWITCH_PERSISTENCE_FRAMES * size_scale)} frames")
            last_logged_size_scale = size_scale
        if consensus_pos is not None:
            consensus_rel = consensus_reliability(n_votes, consensus_spread)
            print(f"Frame {frame_idx} | Reliability | consensus={consensus_rel:.2f} "
                  f"(n_votes={n_votes}, spread={consensus_spread:.1f}px)")
        else:
            consensus_rel = None

        signals_this_frame = []
        if vit_rel is not None:
            signals_this_frame.append(("vit", vit_pos, vit_rel))
        if consensus_rel is not None:
            signals_this_frame.append(("consensus", consensus_pos, consensus_rel))
        if uncertain and gmc_rel is not None and predicted_pos is not None:
            signals_this_frame.append(("gmc", predicted_pos, gmc_rel))

        shadow_pos, shadow_contributors = compute_shadow_fusion(signals_this_frame)
        if shadow_pos is not None:
            contrib_str = ", ".join(f"{name}={weight:.2f}" for name, weight in shadow_contributors)
            print(f"Frame {frame_idx} | SHADOW FUSION | pos=({shadow_pos[0]:.1f},{shadow_pos[1]:.1f}) "
                  f"contributors=[{contrib_str}]")
            n_shadow_fusion_frames += 1
            if len(shadow_contributors) > 1:
                n_shadow_fusion_multi_signal += 1

        if state == "HOLDING":
            final_pos, final_mode, final_detail = held_point, "held", "pre-init"
            sticky_winner, challenger_name, challenger_streak = None, None, 0
            sticky_prev_rel, sticky_decline_streak = None, 0
        else:
            prev_sticky_winner = sticky_winner
            (final_pos, final_mode, final_detail,
             sticky_winner, challenger_name, challenger_streak,
             sticky_prev_rel, sticky_decline_streak) = compute_guarded_final_position(
                signals_this_frame, sticky_winner, challenger_name, challenger_streak,
                sticky_prev_rel, sticky_decline_streak,
                switch_persistence=round(WINNER_SWITCH_PERSISTENCE_FRAMES * size_scale))
            if final_mode == "winner-take-all" and sticky_winner != prev_sticky_winner:
                n_winner_switches += 1
        if final_pos is not None:
            print(f"Frame {frame_idx} | FINAL | pos=({final_pos[0]:.1f},{final_pos[1]:.1f}) "
                  f"mode={final_mode} | {final_detail}")
            if final_mode == "winner-take-all":
                n_winner_take_all_frames += 1

        if (CONSENSUS_GATED_REFRESH_ENABLED
                and state == "TRACKING" and not probation and vit_rel is not None and consensus_rel is not None
                and frames_since_refresh >= TEMPLATE_REFRESH_MIN_COOLDOWN
                and vit_rel >= TEMPLATE_REFRESH_MIN_RELIABILITY
                and consensus_rel >= TEMPLATE_REFRESH_MIN_RELIABILITY):
            agree_dist = math.hypot(vit_pos[0] - consensus_pos[0], vit_pos[1] - consensus_pos[1])
            if agree_dist <= TEMPLATE_REFRESH_AGREEMENT_MAX_DIST:
                refreshed_gray = crop_orb_template(cleaned, bbox)
                if refreshed_gray is not None:
                    recent_orb = orb.detectAndCompute(refreshed_gray, None)
                frames_since_refresh = 0
                n_consensus_gated_refreshes += 1
                print(f"Frame {frame_idx} | CONSENSUS-GATED TEMPLATE REFRESH | vit_rel={vit_rel:.2f} "
                      f"consensus_rel={consensus_rel:.2f} agree_dist={agree_dist:.1f}px "
                      f"(<= {TEMPLATE_REFRESH_AGREEMENT_MAX_DIST})")

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            display_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_window_start = now

    n_active_supporters = sum(1 for s in supporters if s["active"])
    print(f"Summary: {n_transitions} state transitions, {n_lost_frames}/{frame_idx} frames displayed as LOST, "
          f"{n_static_rejections} static-region rejections, {n_distance_rejections} distance-gate rejections, "
          f"{n_retry_cycles} periodic search-retry cycles after abandonment, "
          f"{n_active_supporters}/{len(supporters)} supporters still active at end, "
          f"{n_ambiguous_resolutions} ambiguous re-detections resolved via supporter consensus, "
          f"{n_suspicious_disagreements} suspicious high-confidence disagreements flagged, "
          f"{n_shadow_fusion_frames}/{frame_idx} frames had a computable shadow fusion "
          f"({n_shadow_fusion_multi_signal} from 2+ independent signals), "
          f"{n_winner_take_all_frames} frames the guard picked a single signal over disagreeing others "
          f"({n_winner_switches} actual sticky-winner switches), "
          f"{n_consensus_gated_refreshes} consensus-gated template refreshes")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved output video: {output_path}")


if __name__ == "__main__":
    main()
