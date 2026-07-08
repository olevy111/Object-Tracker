"""Preflight check: verifies this computer can run the tracker.

Run it directly to get a full report:
    python src/check_setup.py

Exits with code 0 if everything needed is in place, 1 otherwise.
The launchers run it automatically after a fresh install.
"""
import os
import platform
import sys

MIN_PYTHON = (3, 9)
MODEL_RELPATH = os.path.join("..", "models", "object_tracking_vittrack_2023sep.onnx")

results = []


def check(name, fn, required=True):
    try:
        detail = fn()
        results.append((True, required, name, detail))
    except Exception as exc:
        results.append((False, required, name, f"{type(exc).__name__}: {exc}"))


def python_version():
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(f"Python {'.'.join(map(str, MIN_PYTHON))}+ required")
    return f"{platform.python_version()} on {platform.system()} {platform.machine()}"


def numpy_import():
    import numpy
    return f"numpy {numpy.__version__}"


def opencv_import():
    import cv2
    return f"opencv-python {cv2.__version__}"


def vittrack_available():
    import cv2
    if not hasattr(cv2, "TrackerVit_create"):
        raise RuntimeError("this OpenCV build has no TrackerVit (need opencv-python >= 4.9)")
    return "cv2.TrackerVit is available"


def model_loads():
    import cv2
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_RELPATH)
    if not os.path.isfile(model):
        raise FileNotFoundError(f"model file missing: {model}")
    params = cv2.TrackerVit_Params()
    params.net = model
    cv2.TrackerVit_create(params)
    return f"model loads ({os.path.getsize(model) // 1024} KB)"


def video_backend():
    import cv2
    cap = cv2.VideoCapture()  # constructing is enough; opening needs a file
    cap.release()
    if not int(cv2.VideoWriter_fourcc(*"mp4v")):
        raise RuntimeError("mp4v codec unavailable")
    return "video read/write backend present"


def gui_backend():
    import cv2
    win = "__preflight__"
    cv2.namedWindow(win)
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return "cv2 window backend works"


def tkinter_available():
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    size = f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}"
    root.destroy()
    return f"tkinter works, screen {size}"


def main():
    print("Pixel Tracker -- setup check\n")
    check("Python version", python_version)
    check("numpy", numpy_import)
    check("OpenCV", opencv_import)
    check("VitTrack support", vittrack_available)
    check("Tracking model", model_loads)
    check("Video backend", video_backend)
    check("Display window", gui_backend)
    check("File dialog (tkinter)", tkinter_available, required=False)

    width = max(len(name) for _, _, name, _ in results)
    failed_required = 0
    for ok, required, name, detail in results:
        if ok:
            mark = "  OK  "
        elif required:
            mark = " FAIL "
            failed_required += 1
        else:
            mark = " WARN "
        print(f"[{mark}] {name.ljust(width)}  {detail}")

    print()
    if failed_required:
        print(f"{failed_required} required check(s) failed -- the app will not run correctly.")
        print("Try deleting the 'venv' folder and running the launcher again.")
        return 1
    if any(not ok for ok, _, _, _ in results):
        print("Ready to run (an optional component is missing; the app has a fallback for it).")
    else:
        print("All checks passed. Ready to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
