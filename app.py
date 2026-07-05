"""Simple launcher UI for the pixel tracker.

Flow:
    1. Enter a video URL or local file path (kept between runs).
    2. The first frame opens -- pick the pixel by mouse click or by typing
       X, ENTER, Y, ENTER to mark, then ENTER again to start tracking.
       Press S at any point to toggle saving the output video.
    3. When the video ends: R restarts from step 1 (address kept), ESC exits.

Usage:
    python app.py            (or double-click run_tracker.bat / run_tracker.command)
"""
import sys

import cv2

from overlay import compute_display_scale
import tracker

WINDOW_NAME = tracker.WINDOW_NAME
MAX_TYPED_DIGITS = 4

KEYS_ENTER = (13, 10)
KEY_ESC = 27
# r / R plus Hebrew-layout resh in every encoding OpenCV may report (unicode/cp1255/cp862)
KEYS_RESET = (ord("r"), ord("R"), 0xE8, 0xF8, 0xA8)
# s / S plus Hebrew-layout dalet in every encoding OpenCV may report
KEYS_SAVE_TOGGLE = (ord("s"), ord("S"), 0xD3, 0xE3, 0x93)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg",
                    ".webm", ".ts", ".flv", ".3gp")
STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")


def clean_source(value):
    """Strip whitespace and surrounding quotes (e.g. from Windows 'Copy as path')."""
    return value.strip().strip('"').strip("'").strip()


def is_video_source(source):
    """Accept only video-file extensions; extension-less stream URLs are allowed."""
    import os
    base = source.split("?")[0].split("#")[0]
    ext = os.path.splitext(base)[1].lower()
    if ext:
        return ext in VIDEO_EXTENSIONS
    return source.lower().startswith(STREAM_PREFIXES)


def prompt_for_video(previous):
    """Small dialog asking for a video URL / local path (with Browse). Returns str or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        while True:
            entered = clean_source(input(f"Video URL or file path [{previous}]: "))
            entered = entered or previous
            if not entered:
                return None
            if is_video_source(entered):
                return entered
            print("Not a video file -- please enter a video (mp4/avi/mov/...) path or URL.")

    result = {"value": None}
    root = tk.Tk()
    root.title("Pixel Tracker - choose video")
    root.geometry("640x150")
    root.resizable(False, False)

    tk.Label(root, text="Enter a video URL / local file path, or browse for a video file:").pack(pady=(12, 4))
    entry_row = tk.Frame(root)
    entry_row.pack(padx=12, fill=tk.X)
    entry = tk.Entry(entry_row)
    entry.insert(0, previous)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    entry.focus_set()
    entry.icursor(tk.END)

    def browse():
        chosen = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", " ".join("*" + e for e in VIDEO_EXTENSIONS))])
        if chosen:
            entry.delete(0, tk.END)
            entry.insert(0, chosen)

    tk.Button(entry_row, text="Browse...", command=browse).pack(side=tk.LEFT, padx=(6, 0))

    def ok(event=None):
        value = clean_source(entry.get())
        if not value:
            return
        if not is_video_source(value):
            messagebox.showerror("Pixel Tracker",
                                 "Only video files are accepted (mp4, avi, mov, mkv, ...).\n"
                                 "Check the address and try again.")
            return
        result["value"] = value
        root.destroy()

    def cancel():
        root.destroy()

    frame = tk.Frame(root)
    frame.pack(pady=10)
    tk.Button(frame, text="Start", width=12, command=ok).pack(side=tk.LEFT, padx=6)
    tk.Button(frame, text="Exit", width=12, command=cancel).pack(side=tk.LEFT, padx=6)
    root.bind("<Return>", ok)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return result["value"]


def show_error(message):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Pixel Tracker", message)
        root.destroy()
    except ImportError:
        print(f"ERROR: {message}")


def draw_label(img, text, org, color, scale=0.65):
    """Text with a black outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_text_banner(img, height, alpha=0.55):
    """Darken a strip at the top of the frame so the help text is readable."""
    banner = img[0:height].copy()
    cv2.rectangle(banner, (0, 0), (img.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(banner, alpha, img[0:height], 1 - alpha, 0, img[0:height])


def select_point_and_options(frame, display_scale, save_video):
    """First-frame selection UI: click or type X/Y, ENTER to mark, ENTER to run.
    S toggles output saving. Returns (point, save_video) or (None, save_video)
    on ESC (back to the video prompt)."""
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

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while picked["point"] is None:
        display = frame.copy()
        draw_text_banner(display, 165)
        draw_label(display, "Click a point, or type X ENTER Y ENTER to mark; ENTER again to start",
                   (20, 40), (0, 255, 255))

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
            status = f"Marked: ({cx}, {cy}) -- ENTER to start tracking, r to reset"
        draw_label(display, status, (20, 75), (0, 200, 255))

        save_label = "ON" if save_video else "OFF"
        save_color = (0, 255, 0) if save_video else (200, 200, 200)
        draw_label(display, f"[S] Save output video: {save_label}", (20, 110), save_color)
        draw_label(display, "[ESC] Back to video selection", (20, 145), (255, 255, 255))

        if current_point is not None:
            cv2.drawMarker(display, clamp_point(current_point), (0, 255, 0),
                            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        shown = (cv2.resize(display, (int(round(display.shape[1] * display_scale)),
                                       int(round(display.shape[0] * display_scale))))
                 if display_scale != 1.0 else display)
        cv2.imshow(WINDOW_NAME, shown)
        key = cv2.waitKey(20) & 0xFF

        if key == KEY_ESC:
            return None, save_video
        elif key in KEYS_SAVE_TOGGLE:
            save_video = not save_video
        elif key in KEYS_RESET:
            pending_click["point"] = None
            typed["x"], typed["y"] = "", ""
            phase["value"] = "typing_x"
        elif key in KEYS_ENTER:
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

    return picked["point"], save_video


def show_end_screen():
    """After the video ends: R = run again (video kept), ESC = exit. Returns True to restart."""
    import numpy as np
    board = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2.putText(board, "Tracking finished.", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(board, "[R]   Run again", (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(board, "[ESC] Exit", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow(WINDOW_NAME, board)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in KEYS_RESET:
            return True
        if key == KEY_ESC:
            return False


def run_tracker(source, point, save_video):
    argv = ["tracker.py", "--video", source, "--x", str(point[0]), "--y", str(point[1])]
    if save_video:
        argv.append("--save-video")
    old_argv = sys.argv
    sys.argv = argv
    try:
        tracker.main()
    finally:
        sys.argv = old_argv


def main():
    source = ""
    save_video = False
    while True:
        entered = prompt_for_video(source)
        if entered is None:
            break
        source = entered

        cap = cv2.VideoCapture(source)
        ok, frame1 = cap.read()
        cap.release()
        if not ok:
            show_error(f"Could not open the video:\n{source}\n\nCheck the address and try again.")
            continue

        display_scale = compute_display_scale(frame1.shape[1], frame1.shape[0])
        point, save_video = select_point_and_options(frame1, display_scale, save_video)
        if point is None:
            cv2.destroyAllWindows()
            continue

        try:
            run_tracker(source, point, save_video)
        except SystemExit:
            pass
        except Exception as exc:
            show_error(f"Tracking stopped with an error:\n{exc}")

        if not show_end_screen():
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
