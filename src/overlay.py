"""
Usage:
    python overlay.py --video "../ex/video.mp4"
    python overlay.py --video "../ex/video.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2
import numpy as np

VX0, VX1 = 240, 1680
CX, CY = 960, 540
CROSSHAIR_ARM = 22
LINE_THICKNESS = 3
DILATE_KERNEL_SIZE = 3
INPAINT_RADIUS = 3

CROSSHAIR_ROI_PAD = 40
FAST_FILL_OFFSET = 6
OBJECT_HINT_HALF_SIZE = 60

BOX_SIZE = 40
WINDOW_NAME = "Stage 1 - Overlay Removal"


def build_overlay_mask(width, height):
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
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.full_mask = build_overlay_mask(width, height)

        x0 = max(0, CX - CROSSHAIR_ARM - CROSSHAIR_ROI_PAD)
        x1 = min(width, CX + CROSSHAIR_ARM + CROSSHAIR_ROI_PAD)
        y0 = max(0, CY - CROSSHAIR_ARM - CROSSHAIR_ROI_PAD)
        y1 = min(height, CY + CROSSHAIR_ARM + CROSSHAIR_ROI_PAD)
        self.roi = (x0, y0, x1, y1)
        self.roi_mask = self.full_mask[y0:y1, x0:x1].copy()

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
            m[y0:y1, x0:x1] = 0
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

    def clean(self, frame, hint_center=None, hint_half_size=OBJECT_HINT_HALF_SIZE):
        out = frame.copy()

        v1 = frame[self.far_y1, self.far_x1].astype(np.int16)
        v2 = frame[self.far_y2, self.far_x2].astype(np.int16)
        out[self.far_ys, self.far_xs] = ((v1 + v2) // 2).astype(np.uint8)

        x0, y0, x1, y1 = self.roi
        roi_crop = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = cv2.inpaint(roi_crop, self.roi_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

        if hint_center is not None:
            hx, hy = hint_center
            ox0 = max(0, int(hx - hint_half_size))
            oy0 = max(0, int(hy - hint_half_size))
            ox1 = min(self.width, int(hx + hint_half_size))
            oy1 = min(self.height, int(hy + hint_half_size))
            if ox1 > ox0 and oy1 > oy0:
                hint_mask = self.full_mask[oy0:oy1, ox0:ox1]
                if hint_mask.any():
                    hint_crop = out[oy0:oy1, ox0:ox1]
                    out[oy0:oy1, ox0:ox1] = cv2.inpaint(hint_crop, hint_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

        return out


def bbox_overlaps_mask(bbox, mask, margin=0):
    x, y, w, h = [int(round(v)) for v in bbox]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(mask.shape[1], x + w + margin)
    y1 = min(mask.shape[0], y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(mask[y0:y1, x0:x1].any())


def draw_box(frame, point, box_size=BOX_SIZE, color=(0, 255, 0)):
    x, y = point
    half = box_size // 2
    cv2.rectangle(frame, (x - half, y - half), (x + half, y + half), color, 2)
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS,
                    markerSize=10, thickness=1)
    return frame


SCREEN_FIT_MARGIN = 0.90


def get_screen_size():
    try:  # Windows
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        pass
    try:  # macOS / Linux
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return 1280, 720


def compute_display_scale(width, height, margin=SCREEN_FIT_MARGIN):
    screen_w, screen_h = get_screen_size()
    return min(1.0, (screen_w * margin) / width, (screen_h * margin) / height)


MAX_TYPED_DIGITS = 4


def pick_point_by_click(frame, display_scale=1.0, window_name=None):
    window_name = window_name or WINDOW_NAME
    picked = {"point": None}
    pending_click = {"point": None}
    typed = {"x": "", "y": ""}
    phase = {"value": "typing_x"}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click["point"] = (x / display_scale, y / display_scale)
            typed["x"], typed["y"] = "", ""
            phase["value"] = "click_pending"

    def clamp_point(pt):
        cx = max(0, min(int(round(pt[0])), frame.shape[1] - 1))
        cy = max(0, min(int(round(pt[1])), frame.shape[0] - 1))
        return cx, cy

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Click a point (ENTER to confirm), or type X, ENTER, type Y, ENTER to mark, "
          "ENTER again to confirm ('r' to reset, ESC to quit)...")
    while picked["point"] is None:
        display = frame.copy()
        cv2.putText(display, "Click+ENTER, or type X ENTER Y ENTER(mark) ENTER(confirm) -- r=reset ESC=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        current_point = None
        if phase["value"] == "click_pending":
            current_point = pending_click["point"]
            cx, cy = clamp_point(current_point)
            status = f"Click pending: ({cx}, {cy}) -- ENTER to confirm, r to reset"
        elif phase["value"] == "typing_x":
            status = f"Type X: {typed['x'] or '_'}  -- ENTER to move to Y, r to reset"
        elif phase["value"] == "typing_y":
            status = f"X={typed['x']}  Type Y: {typed['y'] or '_'}  -- ENTER to mark, r to reset"
        else:
            current_point = (int(typed["x"]), int(typed["y"]))
            cx, cy = clamp_point(current_point)
            status = f"Marked: ({cx}, {cy}) -- ENTER to confirm, r to reset"
        cv2.putText(display, status, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        if current_point is not None:
            cv2.drawMarker(display, clamp_point(current_point), (0, 255, 0),
                            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        shown = (cv2.resize(display, (int(round(display.shape[1] * display_scale)),
                                       int(round(display.shape[0] * display_scale))))
                 if display_scale != 1.0 else display)
        cv2.imshow(window_name, shown)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Selection cancelled by user.")
            sys.exit(0)
        elif key == ord("r"):
            pending_click["point"] = None
            typed["x"], typed["y"] = "", ""
            phase["value"] = "typing_x"
        elif key in (13, 10):  # ENTER
            if phase["value"] == "click_pending":
                picked["point"] = clamp_point(pending_click["point"])
            elif phase["value"] == "typing_x":
                if typed["x"]:
                    phase["value"] = "typing_y"
            elif phase["value"] == "typing_y":
                if typed["y"]:
                    phase["value"] = "marked"
            elif phase["value"] == "marked":
                picked["point"] = clamp_point((int(typed["x"]), int(typed["y"])))
        elif key in (8, 127):  # Backspace
            if phase["value"] == "typing_x":
                typed["x"] = typed["x"][:-1]
            elif phase["value"] == "typing_y":
                typed["y"] = typed["y"][:-1]
        elif ord("0") <= key <= ord("9"):
            if phase["value"] == "typing_x" and len(typed["x"]) < MAX_TYPED_DIGITS:
                typed["x"] += chr(key)
            elif phase["value"] == "typing_y" and len(typed["y"]) < MAX_TYPED_DIGITS:
                typed["y"] += chr(key)

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
