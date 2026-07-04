"""
Stage 4.5 + Stage 5 + Stage 5.5 -- Delayed init near the crosshair,
re-detection (ORB feature matching + static-region guard + velocity-
predicted two-tier search), and context-aware tracking ("supporters") for
disambiguating close, similar-looking objects.

This file is a full copy of stage5_redetection.py (kept as a separate,
independently runnable file per an explicit request, rather than modifying
stage5_redetection.py in place) with Stage 5.5 built on top. See the
"Context-aware tracking" section further down for what Stage 5.5 adds and
why; everything above it is unchanged from stage5_redetection.py.

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

from stage1_overlay import OverlayCleaner, pick_point_by_click, bbox_overlaps_mask, VX0, VX1, CX, CY
from stage3_loss_detection import (
    is_valid, draw_tracking,
    SCORE_THRESHOLD, LOST_N_FRAMES, RECOVER_N_FRAMES, MAX_BBOX_AREA_FRAC, MAX_BBOX_ASPECT_RATIO,
)
from stage4_gmc import build_flow_feature_mask, estimate_motion, GMC_DOWNSCALE

BOX_SIZE = 40
WINDOW_NAME = "Stage 5.5 - Re-detection + Context-Aware Tracking (Supporters)"
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
    if search_gray.shape[0] < MIN_ORB_CROP_SIZE or search_gray.shape[1] < MIN_ORB_CROP_SIZE:
        return None, 0  # ORB's internal pyramid can crash cv2.resize below this size
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


def draw_supporter_selection(frame, target_bbox, supporters, color=(255, 200, 0)):
    """Draw the target box, each supporter, and its offset vector back to
    the target center -- a one-time verification snapshot for Stage 1 (is
    selection picking compact, well-spread structures? see main())."""
    x, y, w, h = [int(v) for v in target_bbox]
    tcx, tcy = x + w // 2, y + h // 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawMarker(frame, (tcx, tcy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    for i, s in enumerate(supporters):
        sx, sy = int(s["pt"][0]), int(s["pt"][1])
        cv2.arrowedLine(frame, (tcx, tcy), (sx, sy), color, 2, tipLength=0.05)
        cv2.circle(frame, (sx, sy), 6, color, 2)
        cv2.putText(frame, f"#{i} [{s['source']}] b={s['bearing']:.0f} deg s={s['strength']:.2f}",
                    (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def draw_supporters_live(frame, supporters):
    """Stage 2: small dot per ACTIVE supporter at its current tracked
    position, brightness = reliability weight -- a continuous, every-frame
    visual check that stable ones keep being tracked while dropped/
    unstable ones disappear (see update_supporters())."""
    for s in supporters:
        if not s["active"]:
            continue
        x, y = int(s["pt"][0]), int(s["pt"][1])
        intensity = int(80 + 175 * min(1.0, s["weight"]))
        cv2.circle(frame, (x, y), 5, (0, intensity, 0), 2)
    return frame


def compute_supporter_consensus(supporters, tolerance=SUPPORTER_VOTE_TOLERANCE,
                                 min_active=SUPPORTER_MIN_ACTIVE_FOR_CONSENSUS):
    """Stage 3: each active supporter votes for the target position (its
    current tracked position minus its fixed init-time offset); the median
    vote is the consensus. A second pass drops any vote further than
    `tolerance` px from that initial median -- absorbs mild affine
    distortion (small rotation/scale drift since init) as ordinary spread
    rather than letting one badly-distorted supporter skew the result (see
    module docstring). Returns (consensus_pos, n_votes_used) or (None, 0)
    if fewer than `min_active` supporters are currently active."""
    active = [s for s in supporters if s["active"]]
    if len(active) < min_active:
        return None, 0

    votes = np.array([(s["pt"][0] - s["offset"][0], s["pt"][1] - s["offset"][1]) for s in active])
    median_vote = np.median(votes, axis=0)
    deviations = np.linalg.norm(votes - median_vote, axis=1)
    inliers = votes[deviations <= tolerance]

    if len(inliers) < min_active:
        # Not enough agreement even with tolerance -- still report the
        # full-set median rather than nothing, but callers can see from
        # n_votes_used that agreement was poor this frame.
        return (float(median_vote[0]), float(median_vote[1])), len(active)

    final = np.median(inliers, axis=0)
    return (float(final[0]), float(final[1])), int(len(inliers))


def draw_consensus(frame, consensus_pos, color=(0, 255, 255)):
    """Small diamond marker at the supporter-consensus predicted position --
    distinct from the purple GMC-motion prediction marker (draw_lost_orb),
    since Stage 3's consensus is an independent signal being validated
    against the tracker's own reported position (see module docstring)."""
    x, y = int(consensus_pos[0]), int(consensus_pos[1])
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_DIAMOND, markerSize=16, thickness=2)
    return frame


def init_supporters_with_preview(gray, init_box, overlay_mask, preview_bgr, static_mask=None, skip_pause=False):
    """Run supporter selection, log the reasoning, and block on a one-time
    annotated preview window so Stage 1's output can be visually verified
    before the run continues (see module docstring). `skip_pause` still
    shows and saves the preview but doesn't block on a keypress -- for
    scripted/batch test runs where nothing is present to press a key."""
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

    preview = preview_bgr.copy()
    draw_supporter_selection(preview, init_box, supporters)
    cv2.putText(preview, "Supporter selection -- press any key to continue", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    preview_window = "Stage 1 - Supporter Selection (verify, then press any key)"
    cv2.imshow(preview_window, preview)
    cv2.waitKey(1 if skip_pause else 0)
    cv2.destroyWindow(preview_window)
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


def draw_holding(frame, held_point, hold_frames, dist_to_crosshair, color=(0, 200, 255)):
    x, y = int(held_point[0]), int(held_point[1])
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
    cv2.putText(frame, f"HOLDING (near crosshair) frame={hold_frames} dist={dist_to_crosshair:.0f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def draw_lost_orb(frame, last_good_bbox, n_matches, mode, predicted_pos=None,
                   color=(0, 0, 255), dim_color=(120, 120, 120), pred_color=(255, 0, 255)):
    if last_good_bbox is not None:
        x, y, w, h = [int(v) for v in last_good_bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), dim_color, 1)
    if predicted_pos is not None:
        px, py = int(predicted_pos[0]), int(predicted_pos[1])
        cv2.drawMarker(frame, (px, py), pred_color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
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
    parser.add_argument("--skip-supporter-preview", action="store_true",
                         help="Don't block on the Stage 1 supporter preview window (for scripted/batch runs)")
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
        point = pick_point_by_click(frame1)
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

        supporters = init_supporters_with_preview(
            cv2.cvtColor(cleaned1, cv2.COLOR_BGR2GRAY), init_box, cleaner.full_mask, cleaned1,
            static_mask=static_mask1, skip_pause=args.skip_supporter_preview)

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

        draw_supporters_live(display, supporters)
        if consensus_pos is not None:
            draw_consensus(display, consensus_pos)

        if state == "TRACKING":
            if probation:
                draw_probation(display, bbox, score, probation_count)
            else:
                draw_tracking(display, bbox, score)
        elif state == "HOLDING":
            draw_holding(display, held_point, hold_frames, math.hypot(held_point[0] - CX, held_point[1] - CY))
        else:
            draw_lost_orb(display, last_good_bbox, match_count_display, search_mode_display, predicted_pos)
            n_lost_frames += 1

        stream_label = "CLEANED" if show_cleaned else "ORIGINAL"
        mask_label = " + MASK" if show_mask else ""
        n_active_supporters = sum(1 for s in supporters if s["active"])
        cv2.putText(display, f"frame {frame_idx}  fps {display_fps:.1f}  [{stream_label}{mask_label}]  "
                              f"supporters {n_active_supporters}/{len(supporters)}",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if consensus_pos is not None:
            cv2.putText(display, f"consensus (yellow diamond) from {n_votes} vote(s)",
                        (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, display)
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
        uncertain = (state == "LOST") or (state == "TRACKING" and probation)
        if uncertain and lost_prev_gray is not None and predicted_pos is not None:
            curr_gray_gmc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_M, _ = estimate_motion(lost_prev_gray, curr_gray_gmc, flow_mask_small)
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
                supporters = init_supporters_with_preview(
                    cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), init_box, cleaner.full_mask, cleaned,
                    static_mask=static_mask_now, skip_pause=args.skip_supporter_preview)
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

                    candidate_bbox, n_matches = None, 0
                    if search_region.shape[0] >= MIN_ORB_CROP_SIZE and search_region.shape[1] >= MIN_ORB_CROP_SIZE:
                        candidate_bbox, n_matches = orb_locate(
                            orb, bf, search_region, [orig_orb, recent_orb],
                            box_size=(args.box_size, args.box_size), offset=(rx0, ry0))
                    match_count_display = n_matches

                    if candidate_bbox is not None:
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
        consensus_pos, n_votes = compute_supporter_consensus(supporters)
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
          f"{n_active_supporters}/{len(supporters)} supporters still active at end")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved output video: {output_path}")


if __name__ == "__main__":
    main()
