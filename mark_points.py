"""
Dev utility -- not part of the staged pipeline. Opens frame 1 of a video and
lets you click points to test later, printing each one's coordinates so you
can hand them back for individual test runs.

Click up to 5 points (numbered as you go). Press 'u' to undo the last click,
'r' to reset, ENTER/SPACE to finish early, q/ESC to quit.

Usage:
    python mark_points.py --video "../ex/track-train.mp4"
"""

import argparse

import cv2

WINDOW_NAME = "Mark points on frame 1 (click up to 5, ENTER when done)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-points", type=int, default=5)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok:
        print("ERROR: could not read first frame.")
        return

    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < args.max_points:
            points.append((x, y))
            print(f"Point #{len(points)}: ({x}, {y})")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"Click up to {args.max_points} points. u=undo  r=reset  ENTER/SPACE=finish  q/ESC=quit")
    while True:
        display = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.drawMarker(display, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            cv2.putText(display, f"#{i + 1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"{len(points)}/{args.max_points} points  |  u=undo r=reset ENTER=done q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            print("Cancelled.")
            return
        if key in (13, 32):  # ENTER or SPACE
            break
        if key == ord("u") and points:
            removed = points.pop()
            print(f"Undid point: {removed}")
        if key == ord("r"):
            points.clear()
            print("Reset.")

    cv2.destroyAllWindows()
    print()
    print("Final points:")
    for i, (x, y) in enumerate(points):
        print(f"  #{i + 1}: --x {x} --y {y}")


if __name__ == "__main__":
    main()
