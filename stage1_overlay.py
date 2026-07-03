"""
Stage 1 -- Overlay removal.

Builds on Stage 0. Removes the drone HUD's fixed crosshair + corner-to-corner
diagonal "X" lines BEFORE the (future) tracker ever sees a frame.

Two frame streams exist every loop:
    - `cleaned`  -> the overlay-removed frame (what the tracker will use later)
    - `original` -> the raw frame (what the user sees by default, so the
                    crosshair/HUD remain visible in the normal view)

Performance note (why two cleaning methods are used):
    A single full-frame cv2.inpaint(TELEA) over this mask costs ~33ms/frame
    (< 30 FPS on its own) because the diagonal lines span corner-to-corner --
    inpaint's cost scales with the region spanned, not the sparse mask area.
    Since the plan's own analysis says the tracked object usually sits under
    the CENTER crosshair (the drone centers its target), we spend the
    accurate-but-costly Telea inpaint only on a small cropped ROI around the
    crosshair (~0.6ms, cheap because the crop is small), and use a fast
    directional gather-scatter fill (sample real pixels from just outside the
    line, perpendicular to it) for the long diagonal lines elsewhere, where
    quality matters less because the object is rarely exactly on those lines.
    Combined cost is ~2.5ms/frame -- ample headroom under the 30 FPS budget.

Keys while playing:
    c  - toggle between showing the ORIGINAL and CLEANED stream
    m  - toggle mask-visualization overlay (mask drawn in red) on top of
         whichever stream is currently shown, to verify coverage
    q / ESC - quit

Usage:
    python stage1_overlay.py --video "../ex/video.mp4"
    python stage1_overlay.py --video "../ex/video.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2
import numpy as np

# ---- Tunable overlay geometry constants (from analysis of the drone HUD) ----
VX0, VX1 = 240, 1680          # video content region x-bounds (rest is black letterbox)
CX, CY = 960, 540             # fixed crosshair center
CROSSHAIR_ARM = 22            # half-length of each crosshair arm (px)
LINE_THICKNESS = 3            # thickness of mask lines before dilation
DILATE_KERNEL_SIZE = 3        # anti-aliasing safety margin around mask lines
INPAINT_RADIUS = 3            # cv2.inpaint radius

CROSSHAIR_ROI_PAD = 40        # extra context (px) around the crosshair arms for the Telea crop
FAST_FILL_OFFSET = 6          # px to sample outside each diagonal line, perpendicular to it

BOX_SIZE = 40
WINDOW_NAME = "Stage 1 - Overlay Removal"


def build_overlay_mask(width, height):
    """Full geometric mask (both diagonals + crosshair), used for
    visualization and as the reference for what "overlay" means."""
    mask = np.zeros((height, width), np.uint8)
    cv2.line(mask, (VX0, 0), (VX1, height), 255, LINE_THICKNESS)
    cv2.line(mask, (VX1, 0), (VX0, height), 255, LINE_THICKNESS)
    cv2.line(mask, (CX - CROSSHAIR_ARM, CY), (CX + CROSSHAIR_ARM, CY), 255, LINE_THICKNESS)
    cv2.line(mask, (CX, CY - CROSSHAIR_ARM), (CX, CY + CROSSHAIR_ARM), 255, LINE_THICKNESS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _dilated_line_mask(height, width, p1, p2):
    m = np.zeros((height, width), np.uint8)
    cv2.line(m, p1, p2, 255, LINE_THICKNESS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
    return cv2.dilate(m, kernel, iterations=1)


class OverlayCleaner:
    """Precomputes everything geometry-dependent once, so per-frame cleaning
    is just cheap array indexing + a small Telea crop."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.full_mask = build_overlay_mask(width, height)

        x0 = max(0, CX - CROSSHAIR_ARM - CROSSHAIR_ROI_PAD)
        x1 = min(width, CX + CROSSHAIR_ARM + CROSSHAIR_ROI_PAD)
        y0 = max(0, CY - CROSSHAIR_ARM - CROSSHAIR_ROI_PAD)
        y1 = min(height, CY + CROSSHAIR_ARM + CROSSHAIR_ROI_PAD)
        self.roi = (x0, y0, x1, y1)
        # Portion of the full mask that falls inside the ROI -- gets high
        # quality Telea inpainting since this is where the target usually is.
        self.roi_mask = self.full_mask[y0:y1, x0:x1].copy()

        # Diagonal-line mask OUTSIDE the ROI -- gets the fast directional fill.
        diag1 = _dilated_line_mask(height, width, (VX0, 0), (VX1, height))
        diag2 = _dilated_line_mask(height, width, (VX1, 0), (VX0, height))
        far_mask = cv2.bitwise_or(diag1, diag2)
        far_mask[y0:y1, x0:x1] = 0

        d1 = np.array([VX1 - VX0, height], dtype=np.float64)
        d1 /= np.linalg.norm(d1)
        n1 = np.array([-d1[1], d1[0]])
        d2 = np.array([VX0 - VX1, height], dtype=np.float64)
        d2 /= np.linalg.norm(d2)
        n2 = np.array([-d2[1], d2[0]])

        ys_list, xs_list = [], []
        y1_list, x1_list, y2_list, x2_list = [], [], [], []
        for line_mask, normal in ((diag1, n1), (diag2, n2)):
            m = line_mask.copy()
            m[y0:y1, x0:x1] = 0  # exclude ROI, handled by Telea instead
            ys, xs = np.where(m > 0)
            sy1 = np.clip(np.round(ys + normal[1] * FAST_FILL_OFFSET).astype(np.int32), 0, height - 1)
            sx1 = np.clip(np.round(xs + normal[0] * FAST_FILL_OFFSET).astype(np.int32), 0, width - 1)
            sy2 = np.clip(np.round(ys - normal[1] * FAST_FILL_OFFSET).astype(np.int32), 0, height - 1)
            sx2 = np.clip(np.round(xs - normal[0] * FAST_FILL_OFFSET).astype(np.int32), 0, width - 1)
            ys_list.append(ys); xs_list.append(xs)
            y1_list.append(sy1); x1_list.append(sx1)
            y2_list.append(sy2); x2_list.append(sx2)

        self.far_ys = np.concatenate(ys_list)
        self.far_xs = np.concatenate(xs_list)
        self.far_y1 = np.concatenate(y1_list)
        self.far_x1 = np.concatenate(x1_list)
        self.far_y2 = np.concatenate(y2_list)
        self.far_x2 = np.concatenate(x2_list)

    def clean(self, frame):
        out = frame.copy()

        # Fast directional fill for the diagonal lines away from the crosshair.
        v1 = frame[self.far_y1, self.far_x1].astype(np.int16)
        v2 = frame[self.far_y2, self.far_x2].astype(np.int16)
        out[self.far_ys, self.far_xs] = ((v1 + v2) // 2).astype(np.uint8)

        # High-quality Telea inpaint restricted to the small crosshair ROI.
        x0, y0, x1, y1 = self.roi
        roi_crop = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = cv2.inpaint(roi_crop, self.roi_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

        return out


def draw_box(frame, point, box_size=BOX_SIZE, color=(0, 255, 0)):
    x, y = point
    half = box_size // 2
    cv2.rectangle(frame, (x - half, y - half), (x + half, y + half), color, 2)
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS,
                    markerSize=10, thickness=1)
    return frame


def pick_point_by_click(frame):
    picked = {"point": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked["point"] = (x, y)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Click a point on the video window to select the object to track...")
    while picked["point"] is None:
        display = frame.copy()
        cv2.putText(display, "Click a point to select the target",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            print("Selection cancelled by user.")
            sys.exit(0)

    return picked["point"]


def main():
    parser = argparse.ArgumentParser(description="Stage 1: overlay removal via mask + inpainting")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--x", type=int, default=None, help="Manual pixel x on frame 1")
    parser.add_argument("--y", type=int, default=None, help="Manual pixel y on frame 1")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE)
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
    mask_pct = 100.0 * np.count_nonzero(cleaner.full_mask) / cleaner.full_mask.size
    print(f"Overlay mask covers {mask_pct:.2f}% of frame")

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

    cv2.namedWindow(WINDOW_NAME)

    show_cleaned = args.show_cleaned
    show_mask = args.show_mask

    frame_idx = 0
    fps_window_start = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    frame = frame1
    while True:
        cleaned = cleaner.clean(frame)
        base = cleaned if show_cleaned else frame
        display = base.copy()

        if show_mask:
            display[cleaner.full_mask > 0] = (0, 0, 255)

        draw_box(display, point, args.box_size)

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

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            display_fps = fps_frame_count / elapsed
            print(f"Frame {frame_idx} | fps={display_fps:.1f}")
            fps_frame_count = 0
            fps_window_start = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
