"""Step 8: the Live Test window (OpenCV). Webcam, or a video file through the same code path.

Preview is mirrored (selfie view) for display only; MediaPipe always runs on the
unmirrored frame (README 6.1). Sign window per README 6.6 (trainer/live.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import recorder
from . import settings as cfg
from .landmarks import create_landmarker, detect, frame_vector
from .live import Classifier, LiveSegmenter, is_correct, save_attempt

WINDOW = "HUDYAT Trainer - Live test   (Q = quit)"
ACCENT = (153, 78, 31)          # BGR of #1F4E99
LEFT_C = (153, 78, 31)          # Left hand: accent blue
RIGHT_C = (0, 106, 178)         # Right hand: amber #B26A00
PANEL_W = 380
FONT_DIR = cfg.PROJECT_DIR / "static" / "fonts"


class Overlay:
    """Side panel drawn with PIL (Poppins), cached until its content changes."""

    def __init__(self, height: int) -> None:
        from PIL import ImageFont
        self.h = height
        self.cache_key = None
        self.cache = None

        def font(name, size):
            try:
                return ImageFont.truetype(str(FONT_DIR / name), size)
            except OSError:
                return ImageFont.load_default()
        self.f_title = font("Poppins-SemiBold.ttf", 22)
        self.f_big = font("Poppins-SemiBold.ttf", 30)
        self.f_body = font("Poppins-Regular.ttf", 17)
        self.f_small = font("Poppins-Regular.ttf", 14)

    def render(self, state: dict) -> np.ndarray:
        key = repr(sorted((k, v) for k, v in state.items() if k != "fps"))
        if key == self.cache_key:
            return self.cache
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (PANEL_W, self.h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        text, muted, line, acc = (26, 26, 26), (107, 107, 107), (230, 230, 227), (31, 78, 153)
        x, y = 24, 22
        d.text((x, y), "Live test", font=self.f_title, fill=text)
        y += 34
        d.text((x, y), f"Model: {state['category']}  ·  {state['mode']} mode", font=self.f_small, fill=muted)
        y += 30
        d.line((x, y, PANEL_W - 24, y), fill=line)
        y += 16
        d.text((x, y), state["status"], font=self.f_body, fill=acc if state.get("signing") else muted)
        y += 36
        if state["mode"] == "Test":
            d.text((x, y), "Target sign", font=self.f_small, fill=muted)
            y += 20
            d.text((x, y), state["target"], font=self.f_big, fill=text)
            y += 46
            d.text((x, y), f"Attempt {min(state['done'] + 1, state['trials'])} / {state['trials']}"
                   f"   ·   correct {state['correct']} / {state['done']}", font=self.f_body, fill=text)
            y += 26
            acc_v = state["correct"] / state["done"] if state["done"] else 0.0
            d.text((x, y), f"Running accuracy {acc_v:.0%}", font=self.f_body, fill=text)
            y += 34
            d.line((x, y, PANEL_W - 24, y), fill=line)
            y += 16
        top3 = state.get("top3") or []
        if top3:
            if not state.get("sure", True):
                d.text((x, y), "Not sure, try again", font=self.f_big, fill=(178, 106, 0))
                y += 48
            else:
                d.text((x, y), top3[0][0], font=self.f_big, fill=text)
                y += 48
            for label, p in top3:
                d.text((x, y), label, font=self.f_body, fill=text)
                d.text((PANEL_W - 24, y), f"{p:.0%}", font=self.f_body, fill=muted, anchor="ra")
                y += 26
                bw = PANEL_W - 48
                d.rectangle((x, y, x + bw, y + 5), fill=(247, 247, 245))
                d.rectangle((x, y, x + int(bw * p), y + 5), fill=acc)
                y += 18
            if state.get("result") is not None:
                y += 6
                good = state["result"]
                d.text((x, y), "Correct" if good else "Wrong", font=self.f_title,
                       fill=(46, 125, 50) if good else (179, 38, 30))
                y += 34
            if state.get("response_ms") is not None:
                d.text((x, y), f"Response time {state['response_ms']:.0f} ms (target < 1000 ms)",
                       font=self.f_small, fill=muted)
        else:
            d.text((x, y), "Raise your hands into view, sign once,", font=self.f_small, fill=muted)
            d.text((x, y + 20), "then lower them out of view.", font=self.f_small, fill=muted)
        d.text((x, self.h - 30), "Q or Esc: quit", font=self.f_small, fill=muted)
        self.cache_key = key
        self.cache = np.asarray(img)[:, :, ::-1].copy()
        return self.cache


def draw_hands(frame_mirrored: np.ndarray, hands) -> None:
    import cv2
    h, w = frame_mirrored.shape[:2]
    for hand in hands:
        color = LEFT_C if hand.label == "Left" else RIGHT_C
        pts = [(int((1 - x) * w), int(y * h)) for x, y, _ in hand.landmarks]   # mirrored for display
        for a, b in recorder.HAND_CONNECTIONS:
            cv2.line(frame_mirrored, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame_mirrored, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_mirrored, p, 3, color, 1, cv2.LINE_AA)
        wx, wy = pts[0]
        cv2.putText(frame_mirrored, hand.label, (wx + 8, wy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv2.LINE_AA)


def open_source(video: str | None, camera: int, backend: str):
    import cv2
    if video:
        cap = cv2.VideoCapture(video)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        return cap, fps, True
    backends = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
    cap = cv2.VideoCapture(int(camera), backends.get(backend, cv2.CAP_MSMF))
    if backend == "dshow":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap, 30.0, False


def run(ctx, category: str, mode: str = "free", tester: str = "", target: str = "", trials: int = 25,
        camera: int | None = None, backend: str | None = None, video: str | None = None,
        headless: bool = False, threshold: float | None = None) -> dict:
    import cv2
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    mdir = paths.model_dir(category)
    if not (mdir / "model.keras").exists():
        raise ds.PlainError(f"No trained model for '{category}'.", "Train it first (step 5).")
    thr = float(threshold if threshold is not None else s.get("confidence_threshold", 0.70))
    rec = recorder.recorder_camera_settings()
    camera = int(camera if camera is not None else (s.get("camera") if s.get("camera") is not None else rec["camera"]))
    backend = backend or s.get("backend") or rec.get("backend", "msmf")
    test_mode = mode == "test"
    if test_mode and not target:
        raise ds.PlainError("Test mode needs a target sign.", "Pick the target sign on the Live Test page.")
    ctx.progress(0.0, "Loading the model...")
    clf = Classifier(mdir, thr)
    if test_mode and target not in clf.labels:
        raise ds.PlainError(f"'{target}' is not a sign of the {category} model.", "Pick a sign from the list.")
    cap, fps, from_file = open_source(video, camera, backend)
    if not cap.isOpened():
        raise ds.PlainError(f"Could not open camera {camera} ({backend}).",
                            "Close Zoom/Teams/the Camera app, or pick another camera or backend in Settings.")
    landmarker = create_landmarker(params=clf.params)
    seg = LiveSegmenter(fps)
    overlay = None
    state = {"category": category, "mode": "Test" if test_mode else "Free", "status": "Waiting for a hand",
             "target": target, "trials": int(trials), "done": 0, "correct": 0, "top3": None, "sure": True,
             "result": None, "response_ms": None, "signing": False}
    attempts = []
    t_start = time.perf_counter()
    frame_i, last_ts = 0, -1
    fps_times: list[float] = []
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:                                            # open in front of the browser
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except (cv2.error, AttributeError):
            pass
    ctx.progress(0.0, "Live test window is open")
    ctx.log(f"Live test: {category}, mode {mode}, source {'video ' + video if from_file else f'camera {camera} ({backend})'}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if from_file:
                    break
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            if from_file:
                ts = int(round(frame_i * 1000.0 / fps))
            else:
                ts = int((now - t_start) * 1000)
                fps_times.append(now)
                fps_times = fps_times[-30:]
                if len(fps_times) >= 10:
                    seg.set_fps((len(fps_times) - 1) / (fps_times[-1] - fps_times[0]))
            ts = max(ts, last_ts + 1)
            last_ts = ts
            frame_i += 1
            hands = detect(landmarker, frame, ts)                      # UNMIRRORED frame
            h, w = frame.shape[:2]
            vec, _ = frame_vector(hands, w, h)
            win = seg.push(vec, len(hands))
            if win is None and from_file and frame_i >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0:
                win = seg.flush()
            state["signing"] = seg.signing
            state["status"] = "Signing..." if seg.signing else "Waiting for a hand"
            result = None
            if win is not None:
                t_end = time.perf_counter()                            # end of the sign window
                result = clf.classify(win)
                state.update(top3=[(l, round(p, 3)) for l, p in result["top3"]], sure=result["sure"],
                             status="Prediction")
                if test_mode:
                    good = is_correct(result["prediction"], result["confidence"], target, thr)
                    state["result"] = good
            display = cv2.flip(frame, 1)                               # mirror for display only
            draw_hands(display, hands)
            if overlay is None or overlay.h != display.shape[0]:
                overlay = Overlay(display.shape[0])
            if result is not None:
                state["response_ms"] = None
            panel = overlay.render(state)
            canvas = np.hstack([display, panel])
            if not headless:
                cv2.imshow(WINDOW, canvas)
            if result is not None:
                response_ms = (time.perf_counter() - t_end) * 1000    # end of window -> prediction shown
                state["response_ms"] = round(response_ms)
                attempt = {"prediction": result["prediction"], "confidence": result["confidence"],
                           "response_ms": response_ms, "frames": result["n_frames"], "reason": win.reason}
                attempts.append(attempt)
                ctx.log(f"attempt {len(attempts)}: {result['prediction']} {result['confidence']:.0%} "
                        f"({', '.join(f'{l} {p:.0%}' for l, p in result['top3'])}), "
                        f"{result['n_frames']} frames, {win.reason}, {response_ms:.0f} ms")
                if test_mode:
                    save_attempt(paths, category, tester, target, result["prediction"], result["confidence"],
                                 state["result"], response_ms)
                    state["done"] += 1
                    state["correct"] += int(state["result"])
                    ctx.progress(state["done"] / max(1, state["trials"]),
                                 f"Attempt {state['done']} / {state['trials']}: {state['correct']} correct")
                    if state["done"] >= state["trials"]:
                        if not headless:
                            cv2.imshow(WINDOW, np.hstack([display, overlay.render(state)]))
                            cv2.waitKey(1500)
                        break
            if not headless:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        landmarker.close()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    msg = f"Live test finished: {len(attempts)} attempt(s)"
    if test_mode and state["done"]:
        msg += f", {state['correct']} correct ({state['correct'] / state['done']:.0%})"
    return {"message": msg + ".", "attempts": attempts, "correct": state["correct"]}
