"""
Stage 0 -- Skeleton.

Load a video, let the user pick a pixel on frame 1 (mouse click, or manual
--x/--y CLI args), draw a fixed-size box around that pixel, and play the
video. No tracking yet -- the box stays put at the clicked location so we
can confirm point selection + playback speed before adding any algorithm.

Usage:
    python stage0.py --video "../ex/video.mp4"
    python stage0.py --video "../ex/video.mp4" --x 960 --y 540
"""

import argparse
import sys
import time

import cv2

# Tunable constants
BOX_SIZE = 40  # side length (px) of the fixed box drawn around the picked point
WINDOW_NAME = "Stage 0 - Skeleton"


def pick_point_by_click(frame):
    """Show `frame` and block until the user left-clicks a point on it."""
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


def draw_box(frame, point, box_size=BOX_SIZE, color=(0, 255, 0)):
    x, y = point
    half = box_size // 2
    cv2.rectangle(frame, (x - half, y - half), (x + half, y + half), color, 2)
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS,
                    markerSize=10, thickness=1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Stage 0 skeleton tracker app")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--x", type=int, default=None, help="Manual pixel x on frame 1")
    parser.add_argument("--y", type=int, default=None, help="Manual pixel y on frame 1")
    parser.add_argument("--box-size", type=int, default=BOX_SIZE,
                         help="Side length in px of the drawn box")
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

    frame_idx = 0
    t_start = time.perf_counter()
    fps_window_start = t_start
    fps_frame_count = 0
    display_fps = 0.0

    frame = frame1
    while True:
        display = frame.copy()
        draw_box(display, point, args.box_size)
        cv2.putText(display, f"frame {frame_idx}  fps {display_fps:.1f}",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

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
