"""
Stage 4.5 + Stage 5 + Stage 5.5 + Stage 5.5.1 -- Delayed init near the
crosshair, re-detection (ORB feature matching + static-region guard +
velocity-predicted two-tier search), context-aware tracking ("supporters")
for disambiguating close, similar-looking objects, and (Stage 5.5.1) a
reliability-weighted fusion layer built on top of all of it.

This file is a full copy of stage5_5_context.py (kept as a separate,
independently runnable file, per the project's established one-file-per-
stage pattern, rather than modifying stage5_5_context.py in place). Stage
5.5.1 supersedes the earlier stage5_6.py fusion experiment -- that file's
approach (guarded correction thresholds, a distinctiveness prior, an
episode-bootstrap prior, all stacked together) got too layered to reason
about and was set aside; this is a deliberately simpler restart.

=== Stage 5.5.1 fusion: guiding principles ===

1. TIME-VARYING RELIABILITY, NO HARD PHASE SWITCH. Signal reliability
   changes over the course of the video -- early on (high altitude, wide
   context, freshly selected supporters on a clear frame) the supporter-
   based consensus position is strong; as the descent progresses,
   supporters leave the frame, scale changes, and consensus reliability
   decays, so appearance (VitTrack/ORB) and motion (GMC) signals become
   relatively more important. The fusion weighting must reflect this
   automatically -- computed FRESH each frame from real, current evidence
   (supporter vote count/agreement, VitTrack's own score, ORB match
   count/spread, GMC's RANSAC inlier ratio) -- never a hard-coded "first N
   seconds vs after" switch, and no artificial bias/prior terms layered on
   top (that was stage5_6.py's approach; not repeated here). The
   supporter-heavy-early-to-appearance-heavy-later shift must EMERGE from
   measurement, not be programmed in directly.

2. NO SIGNAL COUPLING -- SIGNALS MUST NEVER TRAIN OR TEACH EACH OTHER. The
   entire value of fusing multiple independent methods is that they fail
   DIFFERENTLY -- fusion catches one method failing because the others
   don't share its blind spot. Letting one signal's output silently update
   another's internal state (e.g. using supporter consensus positions to
   refresh VitTrack's template or the ORB reference descriptors) destroys
   that independence: a small supporter error gets baked into the
   appearance signal, compounds over subsequent refreshes, and fusion can
   no longer detect the drift because no independent signal is left to
   catch it. Nothing in this file may do that. (The existing periodic
   ORB-template refresh already only ever uses VitTrack's OWN confirmed
   bbox as its source -- appearance refreshing itself, gated on its own
   score -- never supporters. That is fine and is left untouched.)

3. THE ONLY PERMITTED CROSS-SIGNAL BENEFIT: CONSENSUS-GATED TEMPLATE
   REFRESH (future stage, NOT implemented yet in Stage 1 below). The
   appearance template may capture a fresh reference ONLY on frames where
   ALL independent signals (VitTrack, supporter consensus, GMC motion)
   strongly agree on the position -- the agreement itself is the
   verification, not any single signal's say-so. If they don't all agree,
   no update happens. When built, this must be conservative (strong
   agreement only), gated (never from one signal alone), and logged (every
   refresh: which frame, how well the signals agreed).

Stage 5.5.1 build plan (small stages, review after each):
  Stage 1 (this file, so far): measure per-frame reliability for every
    signal + compute a SHADOW fusion (what the blend would produce) purely
    for logging -- nothing here changes the tracker's actual state, the
    displayed box, or any decision. Acting on it is a later stage.
  Stage 2+ (not yet built): guarded acting on the fusion result; the
    consensus-gated template refresh described in principle 3; a single
    unified displayed output.

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
object" behavior of Stages 3/4 with an ACTIVE search while LOST.

REWORKED after a real false-positive bug: whole-frame ORB search was
locking onto random look-alike patches of empty desert far from the real
object (observed: a confirmed "recovery" sitting on blank terrain, nowhere
near the object, with a merely-adequate score). The root cause is that an
appearance-only search considers every location in the frame a candidate --
on repetitive terrain there are always a few look-alike patches that score
decently. The fix is not a better appearance metric; it's to stop
considering far-away locations at all:

1. Motion-predicted search region is now the PRIMARY filter. This footage's
   camera motion is smooth and consistent (measured: ~13.5px/frame average,
   std dx=4.8/dy=10.4, rotation std < 0.5 degrees -- real but bounded
   frame-to-frame variance, not a clean constant velocity). Camera motion is
   estimated every LOST frame via sparse-optical-flow + RANSAC (the Stage 4
   GMC machinery) and used to propagate a predicted object position forward,
   even while nothing is found. ORB search only ever runs inside a bounded
   ROI around that prediction (radius growing with elapsed lost time, capped
   well under frame size) -- there is no whole-frame search tier anymore.
2. Every candidate is hard-gated on distance from the current prediction,
   BEFORE its appearance score is even considered. A match far from where
   the motion model says the object should be is rejected outright,
   regardless of ORB match count -- this is what actually rejects a
   look-alike patch, not the appearance score (measured: unrelated desert
   patches can score plausible ORB match counts too).
3. ORB keypoint/descriptor matching (BFMatcher + knnMatch + Lowe's ratio
   test, ORB_MIN_MATCHES ratio-passing matches required) is still what
   confirms appearance -- ported from old_object_tracker.py's
   ReacquiringTracker (_orb_features / _match / _match_and_locate), which
   measured ORB cross-matches between unrelated desert patches averaging
   ~1.9, far below the 10-match bar, once the candidate pool is small (i.e.
   once the motion gate has already done its job).
4. The static-region (absdiff-vs-previous-frame) guard is unchanged --
   ported from the same reference class (_is_static_region) -- rejects
   pixel-identical-to-last-frame regions (the HUD/crosshair), regardless of
   ORB score.
5. The confidence bar to CONFIRM a recovery (RECOVERY_SCORE_THRESHOLD) is
   set distinctly higher than the bar used to DECLARE loss
   (SCORE_THRESHOLD) -- accepting a recovery at a score barely above the
   lost threshold was too eager.
6. While the predicted position is off-frame (a fully off-screen loss),
   no matching is attempted at all -- there's nothing to find. Search
   resumes, focused near the re-entry edge, only once the prediction
   crosses back into frame bounds.
7. The search is bounded by a TTL (MAX_LOST_FRAMES): if the object hasn't
   been recovered by then, the system stops spending cycles actively
   searching for a while, rather than continuing indefinitely and
   eventually accepting noise -- but this is now a periodic SNOOZE, not a
   permanent give-up (see ABANDON_RETRY_INTERVAL): traced 5 real objects
   that all failed to confirm within one MAX_LOST_FRAMES window and were
   never looked for again for the rest of the clip once abandonment was a
   one-way door, even on this continuously DESCENDING-camera footage where
   the object plausibly becomes easier to re-acquire later (larger,
   clearer) as the drone gets closer. Abandonment now periodically re-
   enables search for one more MAX_LOST_FRAMES-long attempt.

Two saved templates (original anchor, fixed forever from the initial click;
and a periodically refreshed one) are tried on every match attempt, ported
from the same reference class, so a long occlusion or scale change doesn't
strand the system on a stale template while the ground-truth original is
never lost.

A re-detected candidate still starts on PROBATION (see CONFIRM_FRAMES
below) rather than being trusted immediately -- the motion + distance gate
makes false candidates far rarer, but probation is what stops a candidate
that DOES pass every gate from silently overwriting the tracked object's
identity if it doesn't hold up under continued tracking.

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red)
    q / ESC - quit

Usage:
    python stage5_5_context.py --video "../ex/track-train.mp4"
    python stage5_5_context.py --video "../ex/track-train.mp4" --x 960 --y 540
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

# -- ORB matching (ported from old_object_tracker.py's ReacquiringTracker) --
ORB_NFEATURES = 500
ORB_EDGE_THRESHOLD = 8    # default (31) leaves no valid area on a small template -- see module docstring
ORB_PATCH_SIZE = 16
MIN_ORB_CROP_SIZE = ORB_PATCH_SIZE * 2  # below this, ORB's internal pyramid can crash cv2.resize
ORB_RATIO_TEST = 0.75
ORB_MIN_MATCHES = 10
ORB_TEMPLATE_PADDING = 25     # extra context (px) around the tracking box when cropping an ORB template;
                              # a bare 40px box yields ~0-5 keypoints, not enough to ever reach ORB_MIN_MATCHES
TEMPLATE_REFRESH_FRAMES = 30  # refresh the "recent" template every N confident TRACKING frames

# -- Context-aware tracking, Stage 4: multi-candidate re-detection --
# orb_locate_candidates() (below) can report MORE THAN ONE spatially
# distinct match -- the genuinely ambiguous case this whole feature exists
# for (several nearby look-alike objects each independently matching the
# template). Almost always there's still just one.
CANDIDATE_CLUSTER_DISTANCE = 60    # px; ORB matches within this distance of each other are treated as
                                    # the same physical location, not separate candidates
CANDIDATE_MIN_RATIO = 0.5          # a secondary cluster only counts as a genuinely separate candidate
                                    # if its match count is at least this fraction of the top cluster's
                                    # -- a lopsided split (one dominant cluster + a much weaker one) is
                                    # ordinary match noise from a single real object, not a second
                                    # real look-alike (measured this exact false-positive earlier in
                                    # this project's history and had to add this ratio floor to fix it)

# -- Known-obstruction occlusion grace (crossing a fixed overlay line) --
OCCLUSION_MARGIN = 10
# Why: traced a real loss precisely -- score forms a clean V-shape bottoming
# out exactly when the tracked box crosses a fixed overlay line (measured:
# distance-to-line 0.6px, score dropped to 0.225, its minimum in that whole
# stretch), even with the object-following Telea cleaning fix already
# active. Inpainting is always an approximation of the true pixels; VitTrack
# is sensitive enough to notice the difference even when it looks fine to
# the eye. Unlike a generic tracking failure, this is a KNOWN, PREDICTABLE,
# BRIEF event -- we have the exact fixed-line geometry already (the same
# mask Stage 1 inpaints). While the tracked box overlaps that mask (plus a
# small margin), a low score is treated as an expected side effect of the
# crossing rather than evidence of genuine loss: the score-gate is
# suspended, but the geometric sanity checks (size, aspect ratio) are NOT --
# a wildly wrong box is still rejected even mid-crossing. This mirrors
# old_object_tracker.py's Kalman "occlusion bridge" concept (built for the
# crosshair specifically) generalized to the diagonal lines too, without
# needing a full Kalman filter: the tracker's own position is trusted
# through the brief window instead of a separate motion prediction.
#
# Tried and REJECTED: a separate gate rejecting re-detection candidates near
# a fixed overlay line (motivated by a few ACCEPTED candidates during search
# measured 1.4-8.7px from a line). A/B test on two objects showed it had no
# measurable benefit on the hard case it was built for (612/853 LOST either
# way -- that object never re-confirms regardless) while actively breaking a
# different, easier object: a legitimate recovery near a line got rejected,
# nearly 6x'ing that object's LOST time (37/853 -> 218/853). A real object's
# trajectory legitimately passes near these lines just as often as a false
# match does, so proximity-to-line alone isn't a valid re-detection filter --
# only OCCLUSION_MARGIN's grace-for-already-confirmed-tracks approach holds up.

LINE_GRACE_COOLDOWN = 10
# Why a cooldown, not just an instantaneous overlap check: tested the plain
# instantaneous version and the loss STILL fired at the same frame. Tracing
# showed why -- the score doesn't recover the instant the box clears the
# line; it stays depressed for several frames afterward (measured: still
# below the 0.30 lost threshold 7 frames after the box was already 28-70px
# clear of the line). The grace window now stays open for
# LINE_GRACE_COOLDOWN frames after the most recent overlap, covering that
# lingering recovery tail, not just the exact crossing instant.

# -- Re-detection probation (protects the original object's identity) --
CONFIRM_FRAMES = 20
# Why: even with the motion gate below, a candidate that passes every gate
# isn't guaranteed correct. The real risk is a wrong candidate getting
# trusted IMMEDIATELY, overwriting last_good_bbox/pos_history/recent_orb --
# the system's only memory of where and what the real object was. A
# re-detected candidate now starts on PROBATION: the tracker runs on it, but
# nothing is overwritten until it survives CONFIRM_FRAMES consecutive valid
# frames (using RECOVERY_SCORE_THRESHOLD, not the lower LOST threshold). If
# it fails first, it's discarded and the next search resumes from the last
# CONFIRMED position/appearance -- never from the failed guess.

RECOVERY_SCORE_THRESHOLD = 0.42
# Distinctly higher than SCORE_THRESHOLD (0.30, used to DECLARE loss).
# Accepting a recovery at a score barely above the lost threshold (e.g. 0.38)
# was measured to let weak, borderline matches through as if confirmed --
# recovering needs a clearly healthier signal than merely-not-yet-lost.
#
# Tuned down from an initial 0.45 after tracing a genuine (visually verified
# correct, smoothly continuous) recovery that took 226 frames to confirm --
# its score naturally oscillated in the 0.4-0.6 range, and 0.45 meant it
# rarely strung together CONFIRM_FRAMES in a row. At 0.42 the same trace
# would have confirmed by frame 79 instead. This doesn't reopen the door to
# the runaway-box false candidate found in the same test run: that one was
# independently caught by the box-size sanity check (MAX_BBOX_AREA_FRAC)
# regardless of score, so it never had CONFIRM_FRAMES of fully-valid frames
# to begin with at any threshold.

# -- Static-region rejection guard (ported from _is_static_region) --
STATIC_DIFF_THRESHOLD = 2.0
# Calibrated on THIS footage: the on-screen timer region averages ~1.1 mean
# absdiff frame-to-frame (98% of frames below 2.0), while real terrain
# averages ~4.0 (only 18% falsely below 2.0) -- see module docstring. The
# reference implementation used 4.0, calibrated on different footage; kept
# at the value validated against this specific video instead of copied
# blindly, since 4.0 here would falsely reject ~half of real terrain.

# -- Stage 4.5: delayed init near the crosshair --
NEAR_CROSSHAIR_RADIUS = 45   # px; a click within this distance of the crosshair center delays init
                              # (reduced from an initial 80 -- see module docstring: a smaller radius
                              # means a shorter hold, finishing before single-shot LK match quality
                              # degrades over long holds)
HOLD_CORNER_NEIGHBORHOOD = 70  # px radius around the click to search for trackable local corners
HOLD_MAX_CORNERS = 25
MAX_HOLD_FRAMES = 90         # safety cap (~1.5s @ 60fps) so holding never waits forever if the
                              # camera barely moves or GMC repeatedly can't estimate motion

# -- Motion-gated search geometry (replaces the old two-tier local/global design) --
POS_HISTORY_LEN = 5              # tracked positions kept for velocity fallback
SEARCH_BASE_RADIUS = 100         # px, ROI half-width at the moment loss is declared
SEARCH_RADIUS_GROWTH = 6         # px, ROI growth per elapsed LOST frame -- reflects growing
                                  # uncertainty in the propagated motion prediction over time
SEARCH_MAX_RADIUS = 300          # px, ROI half-width cap -- deliberately well under frame size
                                  # (never a whole-frame search, see module docstring point 1)
MAX_ACCEPT_DISTANCE = 320        # px, hard reject for any candidate farther than this from the
                                  # CURRENT motion prediction, independent of appearance score
                                  # (module docstring point 2) -- slightly above SEARCH_MAX_RADIUS
                                  # only to avoid rejecting a legitimate match found right at the
                                  # ROI's own edge/corner
MAX_LOST_FRAMES = 200            # TTL: give up searching gracefully after this many LOST frames
                                  # (module docstring point 7) rather than searching forever
ABANDON_RETRY_INTERVAL = 150     # frames to wait after abandoning before trying again for one more
                                  # MAX_LOST_FRAMES-long window, rather than never again.
                                  # Why: traced 5 real objects that all failed to confirm within the
                                  # first MAX_LOST_FRAMES window and NEVER got looked for again for
                                  # the rest of the clip (600+ frames each) -- abandonment used to be
                                  # a one-way door: both the ORB search and the motion-only fallback
                                  # are gated on `not search_abandoned`, so nothing was ever attempted
                                  # again once tripped, even though this footage is a continuously
                                  # DESCENDING camera -- the object plausibly becomes easier to
                                  # re-acquire later (larger, clearer, less motion blur) as the drone
                                  # gets closer, an opportunity the one-way abandonment could never
                                  # take advantage of. This keeps the original TTL's CPU-saving intent
                                  # (don't spend every single frame searching once a window looks
                                  # hopeless) while periodically checking back in, rather than
                                  # permanently giving up.

# -- Motion-only fallback re-init --
MOTION_FALLBACK_MIN_LOST_FRAMES = 60   # give ORB priority for at least this long first (motion
                                        # prediction drift is smallest early in a lost episode)
MOTION_FALLBACK_INTERVAL = 30          # throttle: only try this once every N lost frames

# Why: traced a real case where the motion prediction stayed in the right
# general neighborhood for the rest of a clip, but ORB never once found
# enough matches there to even propose a candidate -- the object was too
# faint/small, or too visually similar to nearby clutter, for ORB to
# discriminate at that position. If the tracker's OWN regression is more
# capable of fine-tuning onto the right nearby feature than a single-shot
# ORB match (untested going in), trusting the prediction directly, and
# proving it out via the existing PROBATION mechanism (same
# RECOVERY_SCORE_THRESHOLD + CONFIRM_FRAMES bar, same static-region guard on
# the initial candidate), could recover cases ORB alone never even attempts.
# This does NOT skip the safety net -- it only skips the requirement that
# ORB be the one to nominate the candidate.
#
# frames_since_motion_fallback must advance during PROBATION too, not just
# while state=="LOST": traced a real case near a grid line where ORB kept
# proposing short-lived bad candidates (line/inpainting-seam artifacts) that
# each entered TENTATIVE for a few frames before failing -- and the interval
# counter was only wired to increment in the strict LOST branch, so these
# frequent excursions froze it and it never reached MOTION_FALLBACK_INTERVAL.
# The fallback exists precisely to rescue cases like this, so its retry
# clock has to track true wall-clock time since the last attempt, the same
# way lost_frames already does, not just time spent with no candidate at all.

# -- Context-aware tracking, Stage 1: supporter (anchor) selection --
# Concept: even when several nearby objects look visually identical, the
# target's spatial arrangement relative to its neighbors is unique. A
# handful of nearby, generically-distinctive points ("supporters") are
# selected once at init; their fixed offset from the target is the spatial
# fingerprint later stages use to disambiguate re-detection. Selection is by
# GENERIC geometric properties only (corner strength, compactness, spread)
# -- never by absolute appearance (color/brightness) -- so it generalizes to
# any terrain the tracker is tested on, not just this footage's dark-on-pale
# desert structures.
SUPPORTER_SEARCH_RADIUS = 250          # px, neighborhood around the target to look for supporters
SUPPORTER_COUNT = 5                    # target number of supporters (spec range: 3-6)
SUPPORTER_TARGET_EXCLUSION_MARGIN = 10 # px padding around the target box excluded from the search
                                        # (so the target's own corners are never picked as its own supporter)
GOOD_FEATURES_MAX_CANDIDATES = 60      # raw goodFeaturesToTrack corners considered before filtering
GOOD_FEATURES_QUALITY_LEVEL = 0.01     # relative to the strongest corner found in the search region
GOOD_FEATURES_MIN_DISTANCE = 10        # px, minimum separation enforced between raw candidate corners
CORNER_EIGEN_BLOCK_SIZE = 5            # neighborhood size for the structure-tensor (cornerEigenValsAndVecs)
CORNER_EIGEN_KSIZE = 3                 # Sobel aperture size for the same
CORNER_LINE_RATIO_MIN = 0.15           # min(eigenvalue) / max(eigenvalue) of the local structure tensor.
                                        # Near 1.0 = true corner/junction (gradients spread across multiple
                                        # directions); near 0 = a pure edge/line (gradient dominated by ONE
                                        # direction -- the aperture problem: position along the line's own
                                        # length can't be localized). Candidates BELOW this ratio are
                                        # rejected regardless of raw corner strength -- this is the check
                                        # that keeps grid lines and long roads from ever being picked, and
                                        # the single most important knob for generalizing beyond this
                                        # footage (tighten it if a future video's road edges slip through).
SUPPORTER_MIN_BEARING_SEP_DEG = 40      # minimum angular separation (degrees, around the target) enforced
                                        # between chosen supporters -- a good geometric fix needs anchors
                                        # at varied bearings, not all clustered on one side. Selection tries
                                        # this separation first, then relaxes (see select_supporters()) so a
                                        # scene with few strong candidates still yields what it can rather
                                        # than nothing.
SUPPORTER_MIN_BEARING_SEP_FLOOR_DEG = 15  # the relaxation schedule above never goes below this -- see
                                        # select_supporters()
SUPPORTER_MIN_STRENGTH_RATIO = 0.5      # a candidate must be at least this fraction as strong (normalized
                                        # score, relative to the single strongest candidate found in the
                                        # neighborhood) to be eligible at all. Bearing-spread selection only
                                        # searches among these survivors -- it never reaches into a clearly
                                        # weaker candidate just to fill an angular gap or hit SUPPORTER_COUNT.
                                        # Why: traced a real case where the 5-supporter target forced in a
                                        # candidate at 58% of the neighborhood's best strength, sitting on
                                        # unremarkable flat terrain -- not something re-findable later.
                                        # Fewer, genuinely strong supporters beat padding out the count.

# Local-contrast ("blob") candidates -- a SECOND candidate source alongside
# corner detection above. Shi-Tomasi/Harris corner strength measures local
# GRADIENT-DIRECTION diversity at a small pixel neighborhood; it does not
# directly capture "this whole compact patch is much darker/lighter than
# the area around it", which is often the more visually obvious kind of
# distinctiveness (a rock, a bush, a vehicle roof -- anything that stands
# out by contrast against its surroundings, regardless of whether its shape
# happens to be corner-like). Found as local peaks of a band-pass
# (difference-of-blurs) response, which is high at a compact blob of EITHER
# polarity (dark-on-light or light-on-dark) and comparatively low on flat
# regions.
#
# Rejecting lines for THIS channel needs a different test than the corner
# channel's point-wise eigenvalue ratio -- tried that first and it FAILED a
# synthetic check: a smoothly curved blob boundary (a circle) looks locally
# identical to a straight edge at a small pixel neighborhood (gradient
# dominated by one direction, radially), so the point-wise test rejected
# genuine round blobs outright, everywhere along their boundary. A line's
# defining property isn't its local gradient direction, it's that it stays
# elongated over a much LARGER extent than a blob does -- so instead each
# contrast peak gets a REGION-level compactness check (see
# _is_compact_blob()): threshold a local window around the peak, take the
# connected component it belongs to, and reject if that component's
# bounding-box aspect ratio is too elongated, or if it touches the window's
# edge (evidence it actually extends further, i.e. it's a line/edge cut off
# by the window, not a self-contained blob).
CONTRAST_SMALL_KSIZE = 5     # px (odd), the "local patch" blur scale for the contrast map
CONTRAST_LARGE_KSIZE = 25    # px (odd), the "surrounding" blur scale -- the gap between these two
                              # scales sets roughly what blob SIZE the detector is tuned to notice
CONTRAST_QUALITY_LEVEL = 0.2 # peaks below this fraction of the strongest contrast response in the
                              # neighborhood are not even considered candidates
CONTRAST_MIN_DISTANCE = 10   # px, minimum separation enforced between raw contrast-peak candidates
COMPACTNESS_WINDOW_SIZE = 60  # px, local window analyzed around each contrast peak for the
                              # region-based compactness check
COMPACTNESS_MAX_ASPECT = 2.5  # bounding-box max(w,h)/min(w,h) of the peak's connected component --
                              # above this, it's treated as elongated (line/edge-like), not a blob
CANDIDATE_MERGE_DISTANCE = 12  # px; a corner-channel and contrast-channel candidate this close are
                                # treated as the same physical feature -- keep only the stronger one

# Screen-locked HUD exclusion (timer, compass, signal bars, battery, or any
# other on-screen chrome besides the crosshair/grid OverlayCleaner already
# handles). Traced a real case: two "supporters" landed on the on-screen
# "00:15" timer digits -- a static, screen-locked graphic that never moves
# relative to the frame, unlike everything else that moves with the
# camera. Hardcoding this footage's specific HUD element positions would
# defeat the whole point of generalizing to unseen footage with a
# different (or no) HUD layout, so instead this is detected directly:
# compare two consecutive RAW frames, and any pixel that hasn't moved AT
# ALL despite real camera motion elsewhere is almost certainly screen-
# locked, not real scene content. This generalizes to any overlay design.
STATIC_HUD_DIFF_THRESHOLD = 4  # per-pixel absdiff between consecutive frames below this is
                                # considered "hasn't moved" -- see build_static_hud_mask()

# -- Context-aware tracking, Stage 2: tracking supporters frame-to-frame --
# Each supporter chosen at init is tracked every frame via optical flow
# (distinctive corners/blobs are exactly what KLT is good at). A supporter's
# OWN observed motion is cross-checked against the scene's GLOBAL motion
# (the existing GMC machinery, stage4_gmc.estimate_motion) -- ground-fixed
# terrain moves consistently with the camera; something that doesn't (a
# vehicle, a moving shadow, a shifting cloud) is not a valid rigid anchor
# and should stop contributing to the spatial fingerprint. A single
# reliability WEIGHT captures both failure modes (lost tracking and
# GMC-motion mismatch) uniformly: it decays on a bad frame and recovers on
# a good one, so one noisy frame (motion blur, a momentary GMC misfit)
# doesn't instantly kill a supporter, but sustained failure does -- see
# update_supporters(). Leaving the frame is different: not ambiguous, not
# recoverable by waiting, so it's an immediate deactivation, no decay.
SUPPORTER_KLT_WINDOW = 21          # px, optical-flow window size for supporter tracking
SUPPORTER_KLT_MAX_LEVEL = 3        # pyramid levels for the same
SUPPORTER_GMC_MAX_DEVIATION = 15   # px; how far a supporter's OBSERVED motion may differ from what
                                    # the GMC global-scene-motion model predicts for that point before
                                    # it's treated as evidence the supporter isn't ground-rigid
SUPPORTER_TRACK_WEIGHT_DECAY = 0.34    # weight lost on a bad frame (lost track or GMC mismatch) --
                                        # ~3 consecutive bad frames to drop, same rate for both causes
SUPPORTER_TRACK_WEIGHT_RECOVERY = 0.05  # weight regained per good frame -- recovers slower than it
                                        # drops, so a supporter that misbehaved once has to re-earn trust
SUPPORTER_TRACK_MIN_WEIGHT = 0.3   # weight at or below this -- deactivated

# -- Context-aware tracking, Stage 3: consensus position voting --
# Each active supporter "votes" for where the target should be: its current
# tracked position minus its FIXED init-time offset. The median of all
# votes is the consensus -- a robust average, not thrown off by one or two
# outlier votes (a supporter that's drifted, or a point whose local scale
# has changed more than the group's). Because it's a median over whichever
# supporters are currently active, the target position is recoverable from
# any subset -- even 2-3 active supporters still give a usable fix, no need
# for all of them to be present.
#
# The stored offset is a FIXED vector from init -- it does NOT track scale
# or rotation change since init (that would need a similarity-transform fit
# across supporters, not simple per-point voting). Small camera rotation
# (~0.5 deg) and gradual scale change (a descending drone) mean any single
# supporter's vote will drift a little from the "true" arrangement over
# time. Rather than modeling that distortion directly, the SECOND filtering
# pass below tolerates it: votes further than SUPPORTER_VOTE_TOLERANCE from
# the initial group median are treated as outliers (that supporter's
# geometry has distorted too much to trust this frame) and excluded before
# taking the final median -- "mild affine distortion" is absorbed as
# ordinary vote spread rather than requiring an explicit correction.
SUPPORTER_VOTE_TOLERANCE = 40      # px; votes further than this from the initial group median are
                                    # excluded as distorted/outlier before the final consensus
SUPPORTER_MIN_ACTIVE_FOR_CONSENSUS = 2  # fewer active supporters than this -- no consensus attempted
                                    # (spec: even 2-3 give a strong fix, but need at least this many
                                    # to call it a "consensus" at all rather than one lone guess)
SUPPORTER_LINE_EXCLUSION_MARGIN = 8  # px; a supporter currently within this margin of the fixed
                                    # overlay (crosshair/grid lines) is excluded from THIS FRAME's
                                    # vote -- temporary, not a drop. update_supporters() tracks
                                    # supporters on RAW frames (not the cleaned/inpainted ones), so a
                                    # supporter sitting exactly on a line risks KLT matching the
                                    # static, screen-locked graphic itself rather than real scene
                                    # content -- the same class of problem this project has repeatedly
                                    # hit for the TARGET's own tracking near these lines, just not yet
                                    # confirmed as a dominant issue for supporters specifically (checked
                                    # empirically: didn't find obvious corruption in one sample run, but
                                    # this is cheap insurance regardless).

# Periodic supporter refresh (during confirmed TRACKING only): re-run
# selection around the CURRENT trusted position instead of relying forever
# on the init-time fingerprint. Fixes two compounding problems measured
# directly, not just theorized: (1) Stage 2's GMC-rigidity check will
# always eventually drop supporters on real (non-planar) terrain as
# parallax accumulates -- refreshing simply replaces them; (2) even before
# any real drift, the consensus's OWN precision is only ~5-10px (median of
# several individually-noisy KLT votes) -- fine for rejecting a candidate
# hundreds of pixels away, but not always enough to tell apart two lookalike
# objects sitting only 15-30px apart, which is exactly this feature's core
# use case. A stale fingerprint only makes that worse as it drifts further
# from the current true arrangement; a young one (recently refreshed) stays
# as close to that ~5-10px floor as this design can get.
SUPPORTER_REFRESH_INTERVAL = 80    # frames since the last refresh before forcing a new one
SUPPORTER_REFRESH_MIN_ACTIVE = 4   # refresh early if active count drops below this, rather than
                                    # waiting for the interval while running low on supporters
                                    # (tuned from an initial 120/3: measured median consensus-vs-
                                    # tracker distance dropping from 65.5px to 34.4px, and the
                                    # fraction of badly-off frames (>=100px) from 32% to 13%, by
                                    # refreshing more proactively rather than waiting for supporters
                                    # to nearly run out)

# High-confidence disagreement flagging: Stage 4's disambiguation only ever
# runs during a LOST, multi-candidate re-detection -- it never checks a
# CONFIRMED, ongoing track against the supporter consensus. A high-score
# lock onto the wrong nearby look-alike (one that never trips into LOST at
# all) would currently go completely unnoticed, arguably a MORE dangerous
# failure than a low-score wrong lock, since a low-score one usually self-
# corrects into LOST/re-detection where disambiguation can at least try to
# help; a confident wrong lock never gets that chance. This is a
# monitoring signal only for now -- logged, not acted on. Auto-correcting
# on it risks the same class of mistake as the REDETECT_LINE_MARGIN
# regression earlier in this project: acting on an unproven signal can
# break more than it fixes. Log first, decide whether to act once we've
# seen how often and how reliably it actually fires.
SUSPICIOUS_SCORE_THRESHOLD = 0.5    # tracker score at/above this counts as "confident" for this check
SUSPICIOUS_CONSENSUS_DIST = 60      # px; consensus-vs-tracker disagreement at/above this, while
                                    # confident, is flagged (matches CANDIDATE_CLUSTER_DISTANCE's
                                    # sense of "different location", not ordinary vote noise)

# -- Stage 5.5.1, Stage 1: per-signal reliability, measured fresh every
# frame from real evidence -- no prior/bias terms, no phase switch (see
# module docstring principle 1). Values carried over from the stage5_6.py
# experiment where they were empirically calibrated against this footage;
# the calibration itself doesn't depend on that file's fusion-policy layer,
# which is what got dropped, not this measurement layer.
ORB_RELIABILITY_MATCH_SATURATION = 50  # ORB match count at which orb_reliability saturates to 1.0
ORB_RELIABILITY_SPREAD_SCALE = 100     # px; measured n=150 samples across 5 points: min=7.9, p10=22.9,
                                    # p25=31.4, median=48.5, p75=79.3, p90=102.2, max=150.1 -- an
                                    # initial guess of 15 made this saturate to ~0 almost everywhere
CONSENSUS_RELIABILITY_VOTE_SATURATION = SUPPORTER_COUNT
CONSENSUS_RELIABILITY_SPREAD_SCALE = SUPPORTER_VOTE_TOLERANCE
GMC_RELIABILITY_DECAY_HALF_LIFE = 60   # frames; models compounding drift the longer a position
                                    # estimate relies on pure, unanchored motion propagation

# -- Stage 5.5.1, Stage 2: acting on the fusion (guarded) --
# "Confidence-weighted and guarded, never blind averaging" (module
# docstring): blending is only safe when the contributing signals actually
# agree -- averaging genuinely DISAGREEING signals lands on neither
# hypothesis, which is worse than either alone. So: if every pair of
# available signals agrees within FUSION_AGREEMENT_MAX_DIST, blend them
# (weighted by reliability, same as the Stage 1 shadow fusion) -- safe,
# since there's no real outlier to be dragged by. If they disagree beyond
# that, fall back to the SINGLE most reliable signal alone, never a blend.
# No persistence/streak requirement (unlike stage5_6.py's guard): this
# output never overwrites anything permanent (no probation, no
# last_good_bbox) -- it only drives THIS frame's display and logged
# trajectory, so a single frame's own reliability + agreement is enough
# evidence to act on; if it flips back next frame, that's just display
# jitter, not a lasting mistake.
FUSION_AGREEMENT_MAX_DIST = 40  # px; below this, signals are "the same measurement, some noise" and
                                    # safe to blend; at/above, they're plausibly looking at different
                                    # things and blending would be misleading

# Found via the jump-diagnosis testing pass across 7 points: GMC_RELIABILITY_
# DECAY_HALF_LIFE=60 means gmc_rel keeps decaying smoothly toward 0 but never
# hits it EXACTLY -- after a long unanchored LOST stretch (600+ frames,
# observed on real footage when re-detection keeps failing) it becomes a
# vanishingly small positive number (0.00-0.03 displayed). The old "usable if
# rel > 0" filter let that count as a fully legitimate "only signal
# available", so a compounding-drift position thousands of px off-frame got
# displayed as a confident FINAL box -- exactly the failure the user flagged
# earlier ("it will not suddenly show me just similar identifications...
# otherwise it will look like it found it even though it classified a
# different object"). A signal below this floor is now excluded entirely,
# same as if it didn't exist -- if NOTHING clears the floor, final_pos is
# None (no box drawn) rather than a confident wrong answer.
MIN_SIGNAL_RELIABILITY = 0.05

# Found via the same jump-diagnosis testing pass: the winner-take-all branch
# recomputed "who's most reliable" fresh every frame with a plain max(), no
# memory of what it picked last frame. When two signals genuinely disagree
# (a real, different position each) but have closely-matched or noisily-
# fluctuating reliability, the naive winner flip-flops frame to frame --
# observed directly, e.g. gmc/vit trading the lead every 1-2 frames while
# ~330px apart on point (973,533), and the displayed box visibly jumping
# back and forth between the two positions. Fix: give the winner stickiness
# -- once chosen, a challenger must be ahead by a real MARGIN (not just
# numerically higher) AND hold that lead for several consecutive frames
# before the display actually switches. This is the same "clearly better,
# not just marginally, and sustained" guard philosophy as stage5_6.py's
# fusion guard, which worked well there.
WINNER_SWITCH_MARGIN = 0.15       # a challenger must exceed the sticky winner's reliability by at
                                    # least this much -- matches stage5_6.py's FUSION_MIN_RELIABILITY_MARGIN
WINNER_SWITCH_PERSISTENCE_FRAMES = 3  # ...and hold that margin for this many CONSECUTIVE frames before
                                    # the display switches -- a single noisy frame's lead isn't enough

# -- Size-scaled toughness (user observation): as the drone descends, an
# object that's still confirmed and in-frame is BIGGER/closer, which by
# itself means (a) an ordinary few-pixel position disagreement is
# proportionally much less significant on a large box than a small one, and
# (b) a well-established, currently-large confirmed lock deserves more
# sustained proof before being abandoned for a competing signal. This scales
# FUSION_AGREEMENT_MAX_DIST and WINNER_SWITCH_PERSISTENCE_FRAMES continuously
# by the CURRENTLY CONFIRMED object's own measured size -- not by elapsed
# video time or frame count (module docstring principle 1: no hard-coded
# phase switch) -- so it naturally strengthens as the real object grows and
# relaxes back if it doesn't, rather than assuming "later in the video" on
# its own. WINNER_SWITCH_MARGIN (a reliability-space quantity, not a
# spatial one) is deliberately left UNSCALED -- there's no physical
# relationship between object size and how much MORE reliable a challenger
# needs to be, only between size and pixel/frame-count tolerances.
SIZE_SCALE_REFERENCE = BOX_SIZE   # the baseline object size (40px) all scaling is relative to -- a
                                    # confirmed box at this size uses the plain, unscaled thresholds
SIZE_SCALE_MIN = 1.0              # scale never goes below 1.0 -- a SMALLER-than-reference object
                                    # shouldn't become MORE trigger-happy to switch than the tuned defaults
SIZE_SCALE_MAX = 4.0              # cap on how much a large object can inflate the thresholds, so a
                                    # runaway/degenerate oversized box can't make the guard absurdly (or
                                    # unrecoverably) sticky
SIZE_SCALE_EMA_ALPHA = 0.1        # smoothing on the confirmed-size measurement itself (~10-frame time
                                    # constant) -- the box's own size jitters a bit frame to frame even
                                    # while genuinely confirmed; smoothing keeps the scale factor (and
                                    # therefore the effective thresholds) from jittering along with it

# The ONLY permitted cross-signal benefit (principle 3): the appearance
# template may refresh on strong vit+consensus agreement (GMC isn't active
# during confirmed tracking, so it's never part of this particular gate).
# The GATE is mutual agreement; the SOURCE is still VitTrack's own bbox
# (never consensus's position) -- consensus only grants permission, it
# never supplies or teaches the appearance signal anything directly.
TEMPLATE_REFRESH_AGREEMENT_MAX_DIST = 10  # px; tighter than FUSION_AGREEMENT_MAX_DIST -- this gates
                                    # overwriting the appearance reference itself, a bigger deal than
                                    # one frame's displayed position. Tuned down from an initial 20:
                                    # measured directly -- 20px let a "just barely close enough" pair
                                    # through, and the resulting refresh cadence (multiple refreshes
                                    # within a handful of consecutive frames) actively hurt tracking
                                    # (37->242 LOST frames on the same test point) rather than helping,
                                    # by displacing the recent_orb reference so often it stopped acting
                                    # as a distinct, more mature second template.
TEMPLATE_REFRESH_MIN_RELIABILITY = 0.6    # each contributing signal must ALSO be individually STRONG
                                    # (not merely "coincidentally close together"). Measured directly:
                                    # on this footage vit_rel's own 95th percentile is ~0.6 (99th is
                                    # only ~0.64) -- pushing this to a nominally "stronger" 0.85 made it
                                    # unreachable (1 frame out of 763) and inert. 0.6 is genuinely the
                                    # top ~5% for this signal, a fair reading of "strong" for THIS
                                    # footage rather than an arbitrary absolute number.
TEMPLATE_REFRESH_MIN_COOLDOWN = 20  # frames since ANY refresh (periodic or consensus-gated) before a
                                    # new consensus-gated one is allowed -- prevents rapid back-to-back
                                    # refreshes that never let recent_orb settle into a genuinely
                                    # distinct, useful second reference (the direct cause of the
                                    # regression above)
CONSENSUS_GATED_REFRESH_ENABLED = False  # see the DISABLED FOR NOW comment at the call site in main():
                                    # even a single well-gated refresh measurably destabilized
                                    # re-detection later in the video on real footage. Off until a
                                    # safer design is found -- the gate logic below is otherwise ready.


# Deliberately NOT implemented via cap.set(CAP_PROP_POS_FRAMES) peek-then-
# rewind -- tried that first and it silently corrupted playback: seeking on
# this compressed video is not the transparent no-op it looks like (even
# the TOTAL frame count read for the rest of the video changed: 857 vs the
# correct 853), which shifted VitTrack's entire downstream trajectory via
# the same numerical-sensitivity the ORB-candidate-clustering rework hit
# earlier (see the module history). Fixed by never seeking at all: the
# frame read for the HUD-mask comparison is stashed in a `pending_frame`
# slot in main() and consumed by the NEXT normal cap.read() call instead of
# being read twice -- purely sequential decoding throughout.


def build_static_hud_mask(gray1, gray2, threshold=STATIC_HUD_DIFF_THRESHOLD):
    """Pixels that are (near-)identical between two consecutive RAW frames
    despite real camera motion elsewhere -- a screen-locked HUD element,
    not real scene content a supporter could usefully anchor to. Returns a
    uint8 mask (255 = static, to be excluded), or None if `gray2` is
    unavailable (e.g. right at the end of the video)."""
    if gray2 is None:
        return None
    diff = cv2.absdiff(gray1, gray2)
    return (diff < threshold).astype(np.uint8) * 255


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


def estimate_motion_with_inliers(prev_gray, curr_gray, feature_mask_small, scale=GMC_DOWNSCALE):
    """Self-contained duplicate of stage4_gmc.estimate_motion() that
    additionally returns the RANSAC inlier ratio (a rigidity/confidence
    proxy -- see gmc_reliability). Kept as a local copy rather than
    modifying the shared stage4_gmc.py, whose estimate_motion() other
    files already unpack as a plain 2-tuple. Returns (M, n_matches,
    inlier_ratio); M is None and inlier_ratio is 0.0 if too few matches
    survive to trust the estimate."""
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
    """VitTrack's own tracking score, clipped to [0,1] -- already a
    confidence-like value, no rescaling needed."""
    return clip01(score)


def orb_reliability(n_matches, spread):
    """Higher match count and tighter spatial agreement among matched
    keypoints both indicate a more reliable ORB re-detection. Combines
    both into one [0,1] score -- either being poor drags the whole score
    down (min, not average): a huge match count scattered across a wide
    area is not actually reliable, and vice versa."""
    if spread is None:
        return 0.0
    match_score = clip01(n_matches / ORB_RELIABILITY_MATCH_SATURATION)
    spread_score = clip01(1.0 - spread / ORB_RELIABILITY_SPREAD_SCALE)
    return min(match_score, spread_score)


def consensus_reliability(n_votes, spread):
    """Same shape as orb_reliability: more agreeing supporters AND tighter
    agreement among them both raise confidence; either being poor caps it."""
    if spread is None:
        return 0.0
    vote_score = clip01(n_votes / CONSENSUS_RELIABILITY_VOTE_SATURATION)
    spread_score = clip01(1.0 - spread / CONSENSUS_RELIABILITY_SPREAD_SCALE)
    return min(vote_score, spread_score)


def gmc_reliability(inlier_ratio, frames_since_anchor):
    """RANSAC inlier ratio (this frame's own rigidity confidence) decayed
    by how long the current position estimate has relied on pure,
    unanchored motion propagation since the last confirmed anchor --
    compounding drift the longer a position rests on dead reckoning alone."""
    decay = 0.5 ** (frames_since_anchor / GMC_RELIABILITY_DECAY_HALF_LIFE)
    return clip01(inlier_ratio) * decay


def compute_shadow_fusion(signals):
    """Stage 5.5.1, Stage 1: a SHADOW fusion -- the reliability-weighted
    blend of whichever independent position signals are available this
    frame, for LOGGING/inspection only (see module docstring: nothing here
    feeds back into the tracker's state, the displayed box, or any
    decision -- that is a later stage). `signals` is a list of (name, pos,
    reliability) for every signal currently available; entries below
    MIN_SIGNAL_RELIABILITY are dropped entirely (a signal reporting near-
    zero confidence shouldn't dilute ones that aren't, and must never be
    treated as usable just for being "the only one left" -- see
    MIN_SIGNAL_RELIABILITY docstring). Returns (fused_pos, contributors)
    where contributors is [(name, weight_fraction), ...], or (None, [])
    if nothing usable is available."""
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
    """How much to inflate the fusion guard's spatial/persistence thresholds
    given the object's current CONFIRMED size (see SIZE_SCALE_* docstring).
    1.0 at/below the reference size, growing linearly with size above it,
    capped at max_scale."""
    if confirmed_size is None or confirmed_size <= 0:
        return 1.0
    return max(min_scale, min(max_scale, confirmed_size / reference))


def compute_guarded_final_position(signals, sticky_winner, challenger_name, challenger_streak,
                                    max_agreement_dist=FUSION_AGREEMENT_MAX_DIST,
                                    switch_margin=WINNER_SWITCH_MARGIN,
                                    switch_persistence=WINNER_SWITCH_PERSISTENCE_FRAMES):
    """Stage 5.5.1, Stage 2: act on the fusion -- guarded, never blind
    averaging (see FUSION_AGREEMENT_MAX_DIST docstring), and the
    winner-take-all choice is STICKY (see WINNER_SWITCH_MARGIN docstring) --
    once a signal wins a disagreement, another only takes over once it's
    been clearly (not just marginally) ahead for several consecutive
    frames, so two closely-matched disagreeing signals can't flip-flop the
    display frame to frame. `signals` is a list of (name, pos, reliability)
    for every signal available this frame (same shape as
    compute_shadow_fusion's input); `sticky_winner`/`challenger_name`/
    `challenger_streak` are this hysteresis state carried in from the
    previous frame (the caller stores whatever this returns and passes it
    back in next frame). A signal below MIN_SIGNAL_RELIABILITY is excluded
    entirely, same as if it didn't exist.

    Returns (final_pos, mode, detail, new_sticky_winner, new_challenger_name,
    new_challenger_streak). mode is "none", "single", "blended", or
    "winner-take-all"."""
    usable = [(name, pos, rel) for name, pos, rel in signals if rel is not None and rel >= MIN_SIGNAL_RELIABILITY]
    if not usable:
        return None, "none", "", None, None, 0
    if len(usable) == 1:
        name, pos, rel = usable[0]
        return pos, "single", f"only {name} available (rel={rel:.2f})", name, None, 0

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
                None, None, 0)

    by_name = {name: (pos, rel) for name, pos, rel in usable}
    naive_name, naive_pos, naive_rel = max(usable, key=lambda t: t[2])

    if sticky_winner not in by_name:
        # No sticky winner yet, or it's no longer usable this frame (dropped
        # below the floor, or disappeared) -- adopt the naive best directly,
        # nothing stable is being protected.
        return (naive_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- adopting {naive_name} "
                f"(rel={naive_rel:.2f}), no previous sticky winner active",
                naive_name, None, 0)

    sticky_pos, sticky_rel = by_name[sticky_winner]
    if naive_name == sticky_winner:
        return (sticky_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- holding {sticky_winner} "
                f"(rel={sticky_rel:.2f}), the naive best again",
                sticky_winner, None, 0)

    if naive_rel < sticky_rel + switch_margin:
        # A challenger is numerically ahead but not by the required margin --
        # not clearly better, so it doesn't even start a persistence streak.
        return (sticky_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- holding {sticky_winner} "
                f"(rel={sticky_rel:.2f}); {naive_name} (rel={naive_rel:.2f}) ahead but not by the "
                f"{switch_margin} margin required to switch",
                sticky_winner, None, 0)

    new_streak = challenger_streak + 1 if challenger_name == naive_name else 1
    if new_streak >= switch_persistence:
        return (naive_pos, "winner-take-all",
                f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- SWITCHED to {naive_name} "
                f"(rel={naive_rel:.2f}), ahead of {sticky_winner} (rel={sticky_rel:.2f}) by >= "
                f"{switch_margin} for {new_streak} frames",
                naive_name, naive_name, new_streak)
    return (sticky_pos, "winner-take-all",
            f"DISAGREEMENT {max_dist:.1f}px (> {max_agreement_dist}) -- holding {sticky_winner} "
            f"(rel={sticky_rel:.2f}); {naive_name} ahead by >= {switch_margin} for "
            f"{new_streak}/{switch_persistence} frames, not yet enough to switch",
            sticky_winner, naive_name, new_streak)


def cluster_orb_matches(pts, cluster_dist):
    """Greedily group 2D points into spatial clusters by running centroid
    (single-linkage-ish): a point joins the first existing cluster within
    `cluster_dist` of that cluster's current centroid, else starts a new
    one. Cheap and good enough for the small number of matches involved
    here -- the point isn't precise clustering, it's telling apart "one real
    location" from "several distinct look-alike locations"."""
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
    """Search `search_gray` for matches against any of `template_entries`
    (list of (keypoints, descriptors) from orb.detectAndCompute on
    reference templates). Returns a list of (bbox, n_matches, spread) in
    FULL-frame coordinates (using `offset` = the search region's top-left
    in the full frame), sorted by n_matches descending -- `spread` is a
    Stage 5.5.1 reliability signal, see orb_reliability. `box_size` = (w, h) of the
    tracking box to place at each matched location -- NOT derived from the
    spread of matched keypoints (tried and found to blow up unboundedly:
    weak/generic matches scatter across a wide area, and feeding that
    inflated size back into the next template crop compounds into a
    runaway growing box, the same failure shape as the Stage 2 degenerate
    full-frame lock).

    Only the single strongest template's own matches are ever considered:
    a weaker template's result is discarded outright (`continue` on a lower
    match count, never even looking at its position) -- tried keeping every
    template's own qualifying matches and merging across templates by
    spatial proximity instead, and it created FALSE ambiguity: a weaker
    template's own unrelated noise started surviving as a spurious "second
    candidate" whenever it didn't happen to coincide with the stronger
    template's real match. Cross-template disagreement isn't evidence of
    two real look-alike objects; it's just one template being weaker
    (staler, less matched context) than the other.

    Almost always returns exactly one entry -- multiple entries only appear
    when the winning template's OWN matches split into two-plus spatially
    distinct clusters that are each a comparable fraction of the total (>=
    min_ratio of the largest), i.e. that one appearance reference is itself
    genuinely confused by more than one similarly-strong location (see the
    SUPPORTER_* logic in main(), which is what this was built for). A
    lopsided split -- one dominant cluster plus a much smaller one -- is
    just ordinary match noise around a single real object, not a second
    look-alike, and collapses back to a single blended-median candidate."""
    if min_matches is None:
        min_matches = ORB_MIN_MATCHES
    if search_gray.shape[0] < MIN_ORB_CROP_SIZE or search_gray.shape[1] < MIN_ORB_CROP_SIZE:
        return []  # ORB's internal pyramid can crash cv2.resize below this size
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
            continue  # too few matches, or a weaker template than one already seen -- discard
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
        """Mean distance of matched keypoints from their own median --
        Stage 5.5.1 RELIABILITY signal (see orb_reliability): tight
        agreement among matches means high confidence, wide scatter means
        low confidence even at the same match count."""
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


def is_valid_for_recovery(bbox, score, frame_area):
    """Same box-size + aspect-ratio sanity checks as stage3's is_valid(), but
    gated on RECOVERY_SCORE_THRESHOLD instead of the lower SCORE_THRESHOLD
    used to declare loss -- see module docstring point 5."""
    if score < RECOVERY_SCORE_THRESHOLD:
        return False
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if (w * h) > MAX_BBOX_AREA_FRAC * frame_area:
        return False
    return max(w, h) <= MAX_BBOX_ASPECT_RATIO * min(w, h)


def estimate_velocity(pos_history):
    """Velocity (px/frame) from the first and last of the last 3 tracked
    positions just before loss. Returns (0, 0) until enough history has
    accumulated. Used only as a one-frame fallback increment for
    `predicted_pos` on a frame where the GMC motion estimate itself is
    unavailable (too few flow features) -- the primary predictor is the
    continuous GMC propagation in main()."""
    if len(pos_history) < 3:
        return 0.0, 0.0
    (x0, y0), (x1, y1) = pos_history[-3], pos_history[-1]
    return (x1 - x0) / 2.0, (y1 - y0) / 2.0


def transform_point(pt, M):
    """Map a single (x, y) point through affine matrix M (as returned by
    estimate_motion: previous-frame position -> current-frame position)."""
    arr = np.array([[pt]], dtype=np.float32)
    warped = cv2.transform(arr, M)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


def transform_vector(v, M):
    """Map a 2D VECTOR (e.g. a supporter's offset from the target -- a
    DIFFERENCE between two points, not a point itself) through the LINEAR
    part only of affine matrix M, dropping translation. Unlike a point, a
    vector's transform under an affine map has no translation term: if
    both endpoints move by the same M (p1' = A@p1 + t, p2' = A@p2 + t),
    their difference transforms as (p2'-p1') = A@(p2-p1) -- the
    translation cancels out. Used to keep a supporter's stored offset
    valid as the camera's rotation/scale evolves frame to frame, instead
    of treating it as a fixed vector from init (see SUPPORTER_REFRESH_*
    module comment -- measured the fixed-offset assumption producing a
    real, visible, monotonic drift between refreshes)."""
    linear = np.array(M[:, :2], dtype=np.float64)
    dx, dy = linear @ np.array([v[0], v[1]], dtype=np.float64)
    return float(dx), float(dy)


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


def _bearing_deg(dx, dy):
    """Compass-agnostic angle (degrees, [0, 360)) of a 2D offset vector."""
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _circular_sep_deg(a, b):
    """Smallest angular distance between two bearings (degrees, [0, 180])."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _find_contrast_peaks(region, region_mask, min_distance=CONTRAST_MIN_DISTANCE,
                          quality_level=CONTRAST_QUALITY_LEVEL,
                          small_ksize=CONTRAST_SMALL_KSIZE, large_ksize=CONTRAST_LARGE_KSIZE):
    """Polarity-agnostic local-contrast candidates: points where a compact
    patch's intensity differs sharply from its surroundings (a dark OR
    light blob against different-toned surroundings), found as local maxima
    of a band-pass (difference-of-Gaussian-blurs) response. Returns
    (peaks, contrast_map) where peaks is a list of (x, y, response_value)
    in region-local integer coordinates, and contrast_map is the band-pass
    response itself (reused by _is_compact_blob() for the line-rejection
    check on this channel)."""
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
    """Region-based compactness check for a contrast-channel candidate at
    (x, y): threshold a local window around it (relative to its OWN peak
    value, since different peaks sit on different contrast levels), take
    the connected component containing the point, and reject if it's
    elongated (aspect ratio above `max_aspect`) or touches the window's
    edge (evidence it extends further -- a line/edge cut off by the
    window, not a self-contained blob). Returns (is_compact, aspect)."""
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
    """Select context "supporter" anchors around the target at init, by
    GENERIC geometric properties only (never absolute appearance), so this
    generalizes to any terrain:

    1. Raw candidates come from TWO independent sources in a neighborhood
       around the target (overlay mask, `static_mask` -- e.g. a screen-
       locked HUD element, see build_static_hud_mask() -- and the target's
       own box, all excluded from both):
         a. goodFeaturesToTrack (Shi-Tomasi corner strength) -- distinctive
            by local gradient-direction diversity (junctions, corners).
         b. _find_contrast_peaks() -- distinctive by regional intensity
            contrast against surroundings, regardless of polarity (a dark
            or light blob) -- catches compact features a pure corner
            measure can under-rank (e.g. a soft-edged but visually obvious
            blob).
    2. EVERY candidate from EITHER source is independently checked against
       the local structure tensor's eigenvalue RATIO (min/max eigenvalue
       via cornerEigenValsAndVecs) on the ORIGINAL image and rejected if
       below `corner_line_ratio_min` -- a true corner/junction/blob has
       gradients spread across multiple directions (ratio near 1), while a
       point anywhere along an edge or line (a road, a grid line, or the
       body of a high-contrast edge) has gradients dominated by ONE
       direction (ratio near 0) and can't be localized along its own
       length (the aperture problem). This is what rejects long
       roads/lines/edges regardless of which source proposed the point.
    3. Survivors from both sources are merged (a corner and a contrast
       candidate within CANDIDATE_MERGE_DISTANCE of each other are the same
       physical feature -- keep the stronger), each source's raw strength
       normalized to a 0-1 score relative to the strongest survivor IN
       THAT SOURCE first, so neither source's arbitrary numeric scale
       dominates the other.
    4. Candidates below `min_strength_ratio` of the single strongest
       merged candidate are dropped entirely -- prefer fewer, genuinely
       strong supporters over padding out the count with weak filler (see
       SUPPORTER_MIN_STRENGTH_RATIO).
    5. The remaining survivors are greedily selected so the final set is
       spread across bearings around the target (not clustered on one
       side), relaxing the separation requirement in stages if the pool
       can't support it.

    Returns a list of dicts, each: {"pt": (x, y) in full-frame coords,
    "offset": (dx, dy) from the target center, "strength": normalized 0-1
    score, "source": "corner" or "contrast", "ratio": min/max eigenvalue
    (corner-vs-line quality), "bearing": degrees around the target}. May
    return fewer than `target_count` if the scene doesn't have enough
    qualifying candidates (graceful degradation -- see module docstring)."""
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

    # eigen[y, x] = (lambda1, lambda2, x1, y1, x2, y2) -- the two eigenvalues
    # (NOT sorted by OpenCV) and their eigenvectors of the local structure
    # tensor. Only the eigenvalues are needed here.
    eigen = cv2.cornerEigenValsAndVecs(region, blockSize=CORNER_EIGEN_BLOCK_SIZE, ksize=CORNER_EIGEN_KSIZE)

    def eigen_ratio(px, py):
        if not (0 <= px < region.shape[1] and 0 <= py < region.shape[0]):
            return None
        e1, e2 = eigen[py, px, 0], eigen[py, px, 1]
        min_eig, max_eig = (e1, e2) if e1 <= e2 else (e2, e1)
        if max_eig <= 1e-12:
            return None  # flat, textureless region -- not distinctive at all
        return float(min_eig), float(min_eig / max_eig)

    survivors_by_source = {"corner": [], "contrast": []}
    for rx, ry in corner_pts:
        px, py = int(round(rx)), int(round(ry))
        r = eigen_ratio(px, py)
        if r is None or r[1] < corner_line_ratio_min:
            continue  # edge/line-like (aperture problem) -- rejected regardless of strength
        survivors_by_source["corner"].append((rx, ry, r[0], r[1]))
    for px, py, peak_value in contrast_peaks:
        # NOT the point-wise eigenvalue ratio -- see the CONTRAST_* module
        # comment: a smoothly curved blob boundary looks locally identical
        # to a straight edge at a small neighborhood, which made that test
        # reject genuine round blobs. Uses a REGION-level compactness check
        # instead (does the contrast extend in a compact, bounded area, or
        # is it elongated/unbounded like a line?).
        is_compact, aspect = _is_compact_blob(contrast_map, px, py, peak_value)
        if not is_compact:
            continue
        # 1/aspect: same "higher = more point-like" direction as the corner
        # channel's eigenvalue ratio, so both sources log comparably.
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
                "strength": strength / max_strength,  # normalized within its own source, 0-1
                "source": source,
                "ratio": ratio,
                "bearing": _bearing_deg(gx - tcx, gy - tcy),
                "weight": 1.0,      # Stage 2: tracking reliability, see update_supporters()
                "active": True,     # Stage 2: False once dropped (GMC mismatch, lost, off-frame)
            })

    # Corner/contrast candidates landing on the same physical feature --
    # keep only the stronger, rather than treating them as two supporters.
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

    # Relax the separation requirement in stages if the pool can't support
    # the full ask, but NEVER all the way to 0 -- floors out at
    # SUPPORTER_MIN_BEARING_SEP_FLOOR_DEG. Below that, two candidates are
    # close enough in direction to be redundant (measured: two supporters
    # 12px apart, sitting on the same feature, when the schedule used to
    # bottom out at a literal 0-degree requirement). Preferring fewer,
    # genuinely separated supporters over hitting SUPPORTER_COUNT.
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
    """Stage 2: advance each currently-active supporter one frame via
    optical flow, cross-check its motion against the GMC global-scene-
    motion estimate, and update its reliability weight accordingly.
    Mutates `supporters` in place; returns nothing. Logs each individual
    deactivation with its specific cause.

    Three distinct outcomes per supporter, each frame:
      - Left the visible frame (VX0..VX1, 0..height) -- immediate, permanent
        deactivation. Not ambiguous, not something waiting helps with.
      - Optical flow lost track entirely (status=0) -- no new position
        exists to use, so it's left at its last known value; weight
        decays, deactivating once it bottoms out.
      - Flow succeeded -- position ALWAYS updates to what KLT actually
        found (trusting a real measurement over a stale one), but weight
        only recovers if that motion also agrees with the GMC global-
        scene-motion estimate; a mismatch decays weight instead, without
        freezing position -- freezing would penalize a perfectly good,
        ground-rigid supporter for a single noisy GMC estimate by letting
        its tracked position silently drift away from where the point
        actually is. Sustained mismatch (a genuinely non-rigid point, e.g.
        a vehicle) still drops it once weight bottoms out; a one-off
        noisy frame doesn't.

    On a rigid frame, the supporter's stored OFFSET is also updated via
    transform_vector(offset, motion_M) -- not just its position. Measured
    directly (real screenshots + logs from testing): without this, offset
    stays fixed from whenever it was last set (init or refresh) and the
    consensus vote visibly drifts off the true target, monotonically,
    until the next periodic refresh snaps it back. Continuously correcting
    the offset for the same rotation/scale change already being measured
    for the rigidity check keeps the fixed-offset assumption valid on an
    ONGOING basis, not just momentarily after each refresh."""
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
    """Stage 3: each active supporter votes for the target position (its
    current tracked position minus its fixed init-time offset); the median
    vote is the consensus. A second pass drops any vote further than
    `tolerance` px from that initial median -- absorbs mild affine
    distortion (small rotation/scale drift since init) as ordinary spread
    rather than letting one badly-distorted supporter skew the result (see
    module docstring). Returns (consensus_pos, n_votes_used, spread) or
    (None, 0, None) if fewer than `min_active` usable supporters are
    available. `spread` (mean distance of the votes actually used from
    their own median) is a Stage 5.5.1 RELIABILITY signal: tight agreement
    among supporters means high confidence in the consensus, wide
    disagreement means low confidence even with the same vote count.

    If `overlay_mask` is given, any active supporter currently within
    `line_margin` of it is excluded from THIS FRAME's vote (see
    SUPPORTER_LINE_EXCLUSION_MARGIN) -- temporary, not a drop; it resumes
    voting the moment it clears the line."""
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
        # Not enough agreement even with tolerance -- still report the
        # full-set median rather than nothing, but callers can see from
        # n_votes_used (and the now-large spread) that agreement was poor.
        spread = float(np.mean(deviations))
        return (float(median_vote[0]), float(median_vote[1])), len(active), spread

    final = np.median(inliers, axis=0)
    spread = float(np.mean(np.linalg.norm(inliers - final, axis=1)))
    return (float(final[0]), float(final[1])), int(len(inliers)), spread


def init_supporters(gray, init_box, overlay_mask, static_mask=None):
    """Run supporter selection and log the reasoning to the console -- no
    on-screen preview window (see module docstring: the only thing drawn on
    screen is the single final green marker)."""
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


def draw_final_marker(frame, final_pos, box_wh, color=(0, 255, 0)):
    """The ONE on-screen marking of the object's location (module docstring:
    "a single unified displayed output", Stage 2) -- a plain green box
    centered on the Stage 2 guarded final position, sized to VitTrack's own
    box when a valid one is live and current, or a fixed default box
    otherwise (LOST prediction, HOLDING, or a degenerate-VitTrack frame)."""
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

    # Fit the live display to the actual screen instead of always showing the
    # video at its native resolution (which can be bigger than the screen --
    # see module docstring). Purely a DISPLAY concern: all processing/
    # tracking below still runs at full native resolution; only what's shown
    # in the window is scaled, and clicks are mapped back to full-res pixels
    # (see pick_point_by_click).
    display_scale = compute_display_scale(width, height)
    if display_scale < 1.0:
        print(f"Display scaled to {display_scale:.2f}x to fit the screen "
              f"({width}x{height} -> {int(width * display_scale)}x{int(height * display_scale)})")

    ok, frame1 = cap.read()
    if not ok:
        print("ERROR: could not read first frame.")
        sys.exit(1)

    # Static-HUD mask for supporter selection (see build_static_hud_mask()):
    # read sequentially (never seek -- see the comment above
    # build_static_hud_mask()) and stashed so the main loop's first
    # cap.read() consumes this SAME frame instead of skipping past it.
    ok2, pending_frame = cap.read()
    peeked_gray = cv2.cvtColor(pending_frame, cv2.COLOR_BGR2GRAY) if ok2 else None
    if not ok2:
        pending_frame = None
    static_mask1 = build_static_hud_mask(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), peeked_gray)

    if args.x is not None and args.y is not None:
        point = (args.x, args.y)
        print(f"Using manual point: {point}")
    else:
        point = pick_point_by_click(frame1, display_scale=display_scale)
        print(f"Picked point: {point}")

    cleaned1 = cleaner.clean(frame1, hint_center=point)

    half = args.box_size // 2
    params = cv2.TrackerVit_Params()
    params.net = args.model
    orb = make_orb()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

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
    confirmed_object_size = float(BOX_SIZE)
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
        # Screen-fit is a DISPLAY-only concern (see display_scale above) --
        # the saved output video always keeps the full native resolution.
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

        # Position hint for object-following Telea cleaning (see stage1_overlay
        # docstring) -- best-known position as of the END of the previous
        # frame, since this frame's own position isn't known until after
        # cleaning + tracking run on it.
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

        # Stage 2: supporters are tracked every frame regardless of tracking
        # state (distinct from the "uncertain" block below, which only
        # needs GMC to propagate a LOST position estimate) -- a fresh GMC
        # estimate is computed here from consecutive RAW frames (matching
        # how the module already avoids inpainting artifacts in flow
        # features, see flow_mask_small) purely for the rigidity cross-check.
        if supporters:
            prev_frame_gray = cv2.cvtColor(prev_frame_raw, cv2.COLOR_BGR2GRAY)
            curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            supporter_motion_M, _ = estimate_motion(prev_frame_gray, curr_frame_gray, flow_mask_small)
            update_supporters(supporters, prev_frame_gray, curr_frame_gray, supporter_motion_M,
                               width, height, frame_idx)
        prev_frame_raw = frame

        # Motion prediction keeps propagating on EVERY frame we don't yet fully
        # trust -- both while LOST and while a re-detected candidate is still on
        # PROBATION (module docstring point 1) -- so if a tentative lock fails,
        # the next search resumes from an up-to-date prediction, not a stale one.
        gmc_rel = None  # reset every frame -- stays None unless freshly computed just below
        uncertain = (state == "LOST") or (state == "TRACKING" and probation)
        if uncertain and lost_prev_gray is not None and predicted_pos is not None:
            curr_gray_gmc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_M, gmc_n_matches, gmc_inlier_ratio = estimate_motion_with_inliers(
                lost_prev_gray, curr_gray_gmc, flow_mask_small)
            # Stage 5.5.1, Stage 1: GMC reliability -- combines this frame's own
            # RANSAC inlier ratio with a decay based on how long the CURRENT
            # position estimate has relied on pure, unanchored motion
            # propagation (lost_frames resets at each fresh confirmed anchor).
            gmc_rel = gmc_reliability(gmc_inlier_ratio, lost_frames)
            print(f"Frame {frame_idx} | Reliability | gmc={gmc_rel:.2f} "
                  f"(inlier_ratio={gmc_inlier_ratio:.2f}, matches={gmc_n_matches}, "
                  f"frames_since_anchor={lost_frames})")
            if motion_M is not None:
                predicted_pos = transform_point(predicted_pos, motion_M)
            else:
                # GMC couldn't estimate motion this frame (too few flow features) --
                # fall back to a one-frame step of the object's last known velocity
                # rather than freezing the prediction in place.
                vx, vy = estimate_velocity(pos_history)
                predicted_pos = (predicted_pos[0] + vx, predicted_pos[1] + vy)
            lost_prev_gray = curr_gray_gmc

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

                # Sequential read (never seek -- see build_static_hud_mask()
                # comment) stashed in `pending_frame` so the loop's next
                # cap.read() consumes this SAME frame rather than skipping it.
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
            _, bbox = tracker.update(cleaned)
            score = tracker.getTrackingScore()
            valid = is_valid(bbox, score, frame_area)
            # Occlusion grace (see OCCLUSION_MARGIN docstring): a geometrically
            # sane box that's merely low-scoring while crossing a known fixed
            # line is treated as still-tracking, not a failure -- but a box
            # that's ALSO geometrically bogus (size/aspect) never gets this
            # grace, known-obstruction or not.
            geo_sane = (bbox[2] > 0 and bbox[3] > 0
                        and (bbox[2] * bbox[3]) <= MAX_BBOX_AREA_FRAC * frame_area
                        and max(bbox[2], bbox[3]) <= MAX_BBOX_ASPECT_RATIO * min(bbox[2], bbox[3]))
            currently_overlapping = geo_sane and bbox_overlaps_mask(bbox, cleaner.full_mask, OCCLUSION_MARGIN)
            if currently_overlapping:
                line_grace_remaining = LINE_GRACE_COOLDOWN
            elif line_grace_remaining > 0:
                line_grace_remaining -= 1
            near_line = geo_sane and (currently_overlapping or line_grace_remaining > 0)

            if probation:
                lost_frames += 1  # still mid lost-episode until confirmed -- keep the search clock running
                frames_since_motion_fallback += 1  # ditto -- a TENTATIVE excursion must not pause this clock
                if valid:
                    bad_count = 0
                    # Confirmation requires the RAISED bar (module docstring point
                    # 5), not merely the lower threshold that avoids declaring LOST
                    # -- a sane-but-weak frame doesn't advance the confirmation
                    # clock, but doesn't fail the candidate outright either.
                    if is_valid_for_recovery(bbox, score, frame_area):
                        probation_count += 1
                    else:
                        probation_count = 0
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
                        print(f"Frame {frame_idx} | TENTATIVE->LOST (candidate did not hold) | "
                              f"score={score:.3f} | resuming search from last CONFIRMED position")
            elif valid or near_line:
                bad_count = 0
                last_good_bbox = bbox
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                pos_history.append((cx, cy))
                if len(pos_history) > POS_HISTORY_LEN:
                    pos_history.pop(0)
                # Template refresh is still gated on `valid` alone (the ordinary
                # score threshold), not `valid or near_line` -- a low-score frame
                # excused only because of a line crossing can extend tracking,
                # but must never become the new saved appearance template.
                if valid:
                    frames_since_refresh += 1
                    if frames_since_refresh >= TEMPLATE_REFRESH_FRAMES:
                        refreshed_gray = crop_orb_template(cleaned, bbox)
                        if refreshed_gray is not None:
                            recent_orb = orb.detectAndCompute(refreshed_gray, None)
                        frames_since_refresh = 0

                    # Periodic supporter refresh (see SUPPORTER_REFRESH_*
                    # module comment): re-anchor the spatial fingerprint to
                    # the CURRENT confirmed position rather than letting it
                    # grow stale, both from Stage 2 decay and from
                    # accumulated scale/rotation drift since whenever it was
                    # last refreshed.
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
                    # Fully off-frame loss (module docstring point 6) -- nothing to
                    # find while the motion model says the object is off-screen;
                    # keep propagating the prediction but don't spend cycles matching.
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
                        # Genuinely ambiguous: several spatially distinct, appearance-
                        # plausible locations (module docstring: several nearby
                        # look-alike objects). Appearance can't tell them apart --
                        # score each by distance to the supporter consensus (Stage 3)
                        # instead, and take the closest. This is strictly a
                        # disambiguator: the single-candidate path above never
                        # touches supporters at all.
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
                            # Terrain-adaptive fallback: no usable supporter context
                            # (too few/no active supporters) -- fall back to plain
                            # motion-predicted ORB re-detection alone, same as the
                            # unambiguous path, and say so explicitly.
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
                            # Hard motion gate (module docstring point 2) -- rejected
                            # BEFORE appearance is even considered further, regardless
                            # of how many ORB matches it had.
                            n_distance_rejections += 1
                            print(f"Frame {frame_idx} | LOST (distance-gate reject) | matches={n_matches} | "
                                  f"predicted=({pcx:.1f},{pcy:.1f}) | candidate=({ccx:.1f},{ccy:.1f}) | "
                                  f"dist={dist_from_pred:.1f} > {MAX_ACCEPT_DISTANCE}")
                        elif is_static_region(cleaned, prev_cleaned, candidate_bbox):
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
                            line_grace_remaining = 0
                            bbox = candidate_bbox
                            n_transitions += 1
                            print(f"Frame {frame_idx} | LOST->TENTATIVE (re-detected, unconfirmed) | matches={n_matches} | "
                                  f"predicted=({pcx:.1f},{pcy:.1f}) | accepted=({ccx:.1f},{ccy:.1f}) | "
                                  f"dist={dist_from_pred:.1f} | radius={radius:.0f}")

            # Motion-only fallback (see MOTION_FALLBACK_* docstring): ORB gets
            # priority every frame above; only if it still hasn't found/held
            # anything after a while do we periodically trust the motion
            # prediction directly, entering the SAME probation safety net.
            if (state == "LOST" and not search_abandoned and predicted_pos is not None
                    and lost_frames >= MOTION_FALLBACK_MIN_LOST_FRAMES
                    and frames_since_motion_fallback >= MOTION_FALLBACK_INTERVAL):
                frames_since_motion_fallback = 0
                pcx, pcy = predicted_pos
                if VX0 <= pcx <= VX1 and 0 <= pcy <= height:
                    half_b = args.box_size // 2
                    fb_bbox = (int(round(pcx - half_b)), int(round(pcy - half_b)), args.box_size, args.box_size)
                    if is_static_region(cleaned, prev_cleaned, fb_bbox):
                        print(f"Frame {frame_idx} | LOST (motion-fallback static-region reject) | bbox={fb_bbox}")
                    else:
                        tracker = cv2.TrackerVit_create(params)
                        tracker.init(cleaned, fb_bbox)
                        state = "TRACKING"
                        probation = True
                        probation_count = 0
                        bad_count = 0
                        line_grace_remaining = 0
                        bbox = fb_bbox
                        n_transitions += 1
                        print(f"Frame {frame_idx} | LOST->TENTATIVE (motion-only fallback) | "
                              f"predicted=({pcx:.1f},{pcy:.1f}) | bbox={fb_bbox}")

        # Stage 3: consensus-vs-tracker comparison, logged every frame both
        # are available (see module docstring) -- this is purely a
        # VALIDATION signal right now, not yet consulted by the tracker
        # itself (that's Stage 4).
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

        # Stage 5.5.1, Stage 1: per-signal reliability, measured every frame
        # each signal is available -- fresh from real evidence every time,
        # no prior/bias terms (module docstring principle 1). A degenerate
        # (0,0,0,0) VitTrack box (its well-known failure mode) is excluded
        # from ever contributing a "vit" position -- its center would
        # otherwise be a meaningless (0,0), not a real reported position.
        bbox_valid_shape = bbox is not None and bbox[2] > 0 and bbox[3] > 0
        vit_pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) if bbox_valid_shape else None
        if state == "TRACKING" and bbox_valid_shape:
            vit_rel = vit_reliability(score)
            print(f"Frame {frame_idx} | Reliability | vit={vit_rel:.2f} (score={score:.3f})")
        else:
            vit_rel = None

        # Size-scaled toughness (see SIZE_SCALE_* docstring): only a
        # CONFIRMED, trusted box (not probation, not degenerate) updates the
        # reference size -- a momentarily wrong or runaway-oversized
        # tentative box must never inflate how sticky the guard becomes.
        if state == "TRACKING" and not probation and bbox_valid_shape:
            current_size = math.sqrt(bbox[2] * bbox[3])
            confirmed_object_size += SIZE_SCALE_EMA_ALPHA * (current_size - confirmed_object_size)
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

        # Stage 5.5.1, Stage 1: SHADOW fusion -- combine whichever signals
        # are available this frame, weighted by their just-measured
        # reliability. Still logged for comparison against the Stage 2
        # GUARDED result below -- the two differ exactly when signals
        # disagree (module docstring: guarded, never blind averaging).
        # GMC only contributes while its prediction is actively being
        # maintained (uncertain and freshly computed this frame) -- a stale
        # predicted_pos during solid confirmed tracking isn't meaningful.
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

        # Stage 5.5.1, Stage 2: ACT on the fusion -- guarded (see
        # compute_guarded_final_position docstring). This IS the frame's
        # actual output -- what gets displayed and what the logged
        # trajectory records. It does NOT feed back into VitTrack's own
        # tracker state, the ORB templates, or pos_history -- those stay
        # driven by their own original, independent logic (module docstring
        # principle 2); this is a pure OUTPUT layer on top of them.
        if state == "HOLDING":
            final_pos, final_mode, final_detail = held_point, "held", "pre-init"
            sticky_winner, challenger_name, challenger_streak = None, None, 0
        else:
            prev_sticky_winner = sticky_winner
            (final_pos, final_mode, final_detail,
             sticky_winner, challenger_name, challenger_streak) = compute_guarded_final_position(
                signals_this_frame, sticky_winner, challenger_name, challenger_streak,
                max_agreement_dist=FUSION_AGREEMENT_MAX_DIST * size_scale,
                switch_persistence=round(WINNER_SWITCH_PERSISTENCE_FRAMES * size_scale))
            if final_mode == "winner-take-all" and sticky_winner != prev_sticky_winner:
                n_winner_switches += 1
        if final_pos is not None:
            print(f"Frame {frame_idx} | FINAL | pos=({final_pos[0]:.1f},{final_pos[1]:.1f}) "
                  f"mode={final_mode} | {final_detail}")
            if final_mode == "winner-take-all":
                n_winner_take_all_frames += 1

        # Stage 5.5.1, Stage 2: the ONLY permitted cross-signal benefit --
        # consensus-gated template refresh (module docstring principle 3).
        # Strictly ADDITIVE to the existing periodic/score-gated refresh
        # above (unchanged, still self-referential to VitTrack's own box);
        # this is a second, independent opportunity to refresh, available
        # more often early in the video when consensus is strong. The GATE
        # is mutual agreement; the SOURCE is still VitTrack's own bbox,
        # never consensus's position (see TEMPLATE_REFRESH_* docstring).
        #
        # DISABLED FOR NOW (CONSENSUS_GATED_REFRESH_ENABLED = False):
        # measured directly on the (973,533) test point -- even gated on a
        # genuinely strong vit_rel/consensus_rel (~top 5% for this signal),
        # tight 10px agreement, and a 20-frame cooldown, a SINGLE extra
        # refresh at frame 476 preceded a total tracking breakdown by frame
        # 663 (a runaway oversized box, then repeated re-detection matches
        # on the exact same wrong static coordinates) -- 37/853 LOST frames
        # became 209/853. This re-detection pipeline is evidently far more
        # sensitive to recent_orb's exact content than expected; the gate
        # conditions tried so far aren't sufficient to guarantee safety, and
        # further tuning wasn't converging. Left implemented (below) but
        # switched off pending a safer design, rather than shipping it live.
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
