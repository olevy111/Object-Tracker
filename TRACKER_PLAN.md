# Object Tracker — Clean Build Plan (ASIO task)

## Purpose of this document
This is a from-scratch build plan for the ASIO video-tracking task. We are starting the
project **fresh** as a new, clean project. The guiding principle is **small verified steps** —
each stage is built, tested on the real video, and reviewed BEFORE the next stage is added.
We do NOT pile multiple algorithms on at once. Every stage stops for human review.

We carry forward what we LEARNED from earlier experiments, without re-importing the old code
and without getting attached to ideas that didn't pan out.

---

## The task (from the ASIO brief)
Build a Python video-tracking app that:
- Takes a video + a single pixel (x,y) on the first frame (click or manual index) as input.
- Tracks the object at that pixel through the video, drawing its tracked location + a
  bounding box (box size around the pixel is our choice).
- Handles **losing** the object and **re-acquiring** it when it re-enters the frame.
- Runs on a **standard CPU — NO GPU** — in **real time: >= 30 FPS at 1920x1080**.
- Must work on ANY clearly-visible object the user picks (tested on separate videos we
  won't see in advance).
- External libraries (OpenCV etc.) are allowed.

Deliverable: public GitHub repo + README + short demo video + a written summary answering:
steps taken, guiding principles, algorithms explored & conclusions, hardest challenge,
where it's limited.

---

## What we learned from earlier experiments (carry this knowledge forward)

**The video is a drone feed with a fixed HUD overlay.** Confirmed by analysis of the actual
training video (`אימון_עקיבה.mp4`, 1920x1080):
- Video region is x ∈ [240, 1680] (width 1440); the rest is **black letterbox bars**.
- A fixed **white crosshair** sits at the exact center (960, 540).
- Two fixed **diagonal "X" lines** run corner-to-corner of the video region (~37° angle,
  passing through center).
- A fixed **HUD strip** at the bottom (speed, RC/HD, Mbps, battery %), a **timer** upper-right
  (digits change, position fixed), and an occasional **red recording dot**.
- The overlay is IDENTICAL across all videos from this drone (same system).

**Findings that shape the design:**
1. **The camera moves a lot** (falling/panning drone) — the ground shifts significantly
   between frames while the overlay stays fixed. This was the main cause of tracking loss.
2. **The object the user picks is usually right under the center crosshair** (the drone
   centers its target). So the crosshair directly overlaps the object — this MUST be handled.
3. **The target objects are often tiny** (a few dozen pixels — as small as the crosshair).
   This means overlay handling must be surgical: every wrongly-masked pixel destroys part of
   a tiny object.
4. **FPS was never the problem** — earlier runs hit 130+ FPS at 1080p. We have huge headroom.
   The problem was always **accuracy / drift**, not speed.
5. **Piling on components broke things.** Skipping straight to fancy trackers or extra modules
   without a validated confidence signal led to a system where "everything failed" and the
   cause was un-diagnosable. Hence: small steps, measure each one.

**Overlay-cleaning solution (already validated, use as the Stage 1 basis):**
We remove the overlay BEFORE the tracker sees each frame, using a **precise geometric mask +
Telea inpainting**. This was tested on the real video and works cleanly — the crosshair and
X-lines disappear and the ground is reconstructed smoothly. Key parameters that worked:

```python
# Frame is 1920x1080. Video region x in [240,1680]. Center (960,540).
VX0, VX1 = 240, 1680
CX, CY = 960, 540
mask = np.zeros((1080,1920), np.uint8)
cv2.line(mask, (VX0,0), (VX1,1080), 255, 3)   # X diagonal 1 (corner-to-corner of video)
cv2.line(mask, (VX1,0), (VX0,1080), 255, 3)   # X diagonal 2
cv2.line(mask, (CX-22,CY), (CX+22,CY), 255, 3) # crosshair horizontal arm
cv2.line(mask, (CX,CY-22), (CX,CY+22), 255, 3) # crosshair vertical arm
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)  # 2px AA catch
cleaned = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```

This mask is ~1.2% of the frame — tight, thin, crosshair drawn as a cross (not a box).
The HUD strip / timer / red dot can be added to the mask the same way if they interfere
(they're away from center so lower priority). Make ALL geometry constants tunable at the top
of the file, and provide a **mask-visualization mode** (draw mask in red on a frame) to verify
coverage before wiring into the pipeline.

---

## The target architecture (built in stages, not all at once)
1. **Overlay removal** (geometric mask + Telea inpaint) — tracker only ever sees cleaned frames.
2. **Short-term core tracker** — follows the object frame-to-frame; exposes a confidence score.
3. **Loss detection** — score-gated (NOT the tracker's own `ok` flag, which we found unreliable);
   with hysteresis so it doesn't flicker.
4. **Global Motion Compensation (GMC)** — cancels the drone's camera motion each frame so the
   tracker searches in the right place. Uses sparse optical flow + RANSAC affine; MUST exclude
   the overlay-mask region from feature selection (fixed lines would fool the motion estimate).
5. **Re-detection** — when lost, search to re-lock when the object reappears.

**Tracker choice:** start with an OpenCV built-in. VitTrack (`TrackerVit`) is CPU-capable and
exposes a confidence score — good default. CSRT/KCF are fine as a Stage-2 baseline but have no
re-detection on their own. Do NOT use GPU-only models (SAM2, CoTracker — both need CUDA and,
for CoTracker, a non-commercial license). Point tracking / heavy transformers are out for the
CPU + real-time constraint.

---

## STAGED PLAN — build ONE stage, test on the real video, STOP for review

### Stage 0 — Skeleton
Minimal app: load video, show it in a window, let the user click a pixel on frame 1 (and also
accept manual (x,y) input), draw a fixed-size box around it. No tracking yet. Print frame
number + FPS. Confirm we can select a point and that the video plays at >=30 FPS.
**Stop for review.**

### Stage 1 — Overlay removal (the thing that blocked us — solve it first)
Add the geometric mask + Telea inpainting above. Two frame streams: **cleaned** (for the
tracker, later) and **original** (for display, so the user still sees the crosshair/HUD).
Add a mask-visualization mode. Verify on the real video that the crosshair and X-lines are
cleanly removed and the ground looks natural, especially at the center where tiny objects live.
**Stop for review.**

### Stage 2 — Short-term core tracker
Initialize the tracker on the **cleaned** frame at the clicked pixel (NEVER the original —
the template must be crosshair-free). Feed **cleaned** frames to every update. Draw the box +
print the confidence score every frame. Test while the object stays visible. Confirm the box
holds on the object and confidence stays healthy. **Stop for review.**

### Stage 3 — Loss detection (score-gated + hysteresis)
Declare LOST when the confidence score stays below a tunable threshold for N consecutive
frames (start ~0.30 / N=3), regardless of the tracker's `ok` flag. Require the score above
threshold for a couple of frames before returning to TRACKING. Log every state transition
(`Frame X | TRACKING->LOST | conf ...`). Mark stale scores clearly during LOST/SEARCH states.
Test on a clip where the object leaves the frame — confirm LOST fires at the right moment.
**Stop for review.**

### Stage 4 — Global Motion Compensation
Estimate camera motion between consecutive frames (`cv2.goodFeaturesToTrack` +
`cv2.calcOpticalFlowPyrLK` + `cv2.estimateAffinePartial2D`). **Exclude the overlay-mask region
from feature detection** so the fixed lines/crosshair aren't picked as motion anchors. Use the
estimated motion to predict the object's expected position each frame BEFORE the tracker runs,
so its search starts from the motion-compensated location. Measure FPS (we have headroom).
Test: does confidence now hold through the camera motion instead of decaying? **Stop for review.**

### Stage 4.5 — CONDITIONAL: delayed-init for tiny objects under the crosshair
**Only build this if the data shows it's needed** — i.e. if, after Stages 1–4, tiny objects
that sit directly under the center crosshair still can't be locked onto because inpainting
removes too much of them.

The problem: the drone centers its target, so the user's clicked object is usually right under
the crosshair. If the object is large enough, inpainting the thin crosshair leaves enough of it
to track (no special handling needed). But if the object is as small as the crosshair itself,
inpainting erases it — there's no template to lock onto.

The solution (delayed initialization): detect when the click is **near the center crosshair**
AND the object is tiny. In that case, don't initialize immediately. Instead:
1. Hold the clicked pixel position.
2. Use the **GMC motion estimate** (from Stage 4) to track where that ground point moves as the
   camera shifts, frame by frame.
3. Once the camera has moved enough that the object has slid **out from under the crosshair** and
   sits on clean ground, grab a fresh template there and initialize the tracker normally.
4. From that point on, proceed with the standard tracking logic (as if the click had been on a
   clear object away from the crosshair).

If the click is **far from the crosshair/lines** to begin with, skip all of this — initialize
immediately on the cleaned frame (the normal path).

This stage is explicitly LAST-RESORT and depends on GMC. Do not build it preemptively. First
confirm with the real video whether inpainting alone (Stages 1–2) already handles the tiny
under-crosshair object. **Stop for review before building, and only build if the data calls for it.**

### Stage 5 — Re-detection
When Stage 3 declares LOST, switch to re-detection: save a template from the first frame (and
a few high-confidence frames), and search to re-lock when the object reappears. Prefer
efficient search (e.g. random-search / template matching) over brute-force sliding window.
Re-lock and resume tracking when a confident match is found. **Stop for review.**

### Stage 6 — Robustness, README, demo
Tune thresholds across the different video regions, handle edge cases, confirm the end-to-end
FPS budget, write the README + summary answers, record the demo video.

---

## Working rules (important — this is how we avoid the earlier mess)
- **One change at a time, then measure.** No stacking multiple new components in one step.
- **CPU-only, mind the >=30 FPS @ 1080p budget** at every stage; flag any drop.
- **Diagnose with data, don't guess.** When something fails, add visibility (log the numbers)
  before changing logic. The confidence score is the key signal — always be able to see it.
- **Use what we learned, but don't fall in love with any idea.** If a stage's data says an
  approach isn't working, we change course based on the data.
- **Small, reviewable diffs.** Explain what changed and why after each stage.
- **Stop after each stage** and wait for the human to review and direct the next step.
- If existing code or a proposed change conflicts with this plan, say so before proceeding.

Start with **Stage 0**.
