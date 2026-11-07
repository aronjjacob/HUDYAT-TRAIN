"""Step 5: the Live test window, in the style of the data collector's recording screen (trainer/live_ui.py).

The camera fills the window as a mirrored selfie view; MediaPipe runs on the unmirrored frame (README 6.1). Sign
windows per README 6.6 (trainer/live.py). Recognizer "trainer": the model is used exactly as its preprocess.json says
(canonical hand with the Hand setting Right / Left / Auto). Recognizer "app": the HUDYAT app's exported model files,
with the recommended capture ("fixed") or the app's capture today ("app"), see trainer/live.py. A recorded video goes through the same code path with
the hidden --video option (automated tests), and --screenshot DIR saves the Free, Test and summary screens.

Keys: Space start / pause, N next sign, H hand, M mode, Q / Esc quit. Every key also has a button.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import live_ui as ui
from . import recorder
from . import settings as cfg
from .landmarks import create_face_landmarker, create_landmarker, detect, detect_face
from .live import CAPTURES, AppClassifier, AppClipSegmenter, Classifier, LiveSegmenter, save_attempt
from .live_session import LiveSession

WINDOW = "HUDYAT Trainer - Live test"
TONE = {"ok": ui.SUCCESS, "unsure": ui.WARNING, "wrong": ui.DANGER}
HAND_TEXT = {"right": "Right", "left": "Left", "auto": "Auto"}
M = ui.MARGIN


def pretty(sign: str) -> str:
    return sign.replace("_", " ").title()


def _with_badge(canvas, hit, sprite, x, y, key: str, letter: str | None) -> int:
    """Draw a pill / button at (x, y), register it as clickable, add its key badge. Returns its right edge."""
    sprite.draw(canvas, x, y)
    hit.add(int(x), int(y), sprite.width, sprite.height, key)
    if letter:
        b = ui.key_badge(letter)
        b.draw(canvas, x + sprite.width - b.width + 6, y - 9)
    return int(x + sprite.width)


def draw_screen(frame, hands: list, sess: LiveSession, info: dict) -> tuple[np.ndarray, ui.HitMap]:
    """The whole window for one frame. info: category, hands_seen (bool), message (str, optional)."""
    hit = ui.HitMap()
    if frame is None:
        canvas, rect = ui.blank_canvas()
    else:
        canvas, rect = ui.compose_video(frame, mirror=True)
        ui.draw_face_points(canvas, info.get("face"), rect, mirrored=True)
        ui.draw_hands(canvas, hands, rect, mirrored=True)
    test = sess.mode == "test"

    # --- top-left: back, category, mode, hand, hands-seen -----------------------------------------
    x, y = M, M
    x = _with_badge(canvas, hit, ui.round_button("back", 44), x, y - 2, "quit", "Q") + 12
    cat = ui.pill(ui.pretty_category(info["category"]), 16, "semibold", height=40)
    cat.draw(canvas, x, y)
    x += cat.width + 10
    mode = ui.pill("Test" if test else "Free", 16, "semibold", bg=ui.PRIMARY if test else ui.DARK,
                   bg_alpha=0.9 if test else ui.PANEL_ALPHA, height=40)
    x = _with_badge(canvas, hit, mode, x, y, "mode", "M") + 14
    hand = ui.pill(f"Hand: {HAND_TEXT.get(sess.hand, sess.hand)}", 16, "medium", height=40)
    x = _with_badge(canvas, hit, hand, x, y, "hand", "H") + 14
    seen = info.get("hands_seen", False)
    hp = ui.pill("Hands", 15, "medium", dot=ui.SUCCESS if seen else ui.MUTED, height=40)
    hp.draw(canvas, x, y)
    if info.get("engine"):
        ui.pill(info["engine"], 14, "medium", height=40).draw(canvas, x + hp.width + 14, y)

    # --- Test mode: target, trial progress, running accuracy ----------------------------------------
    if test and sess.target:
        name = pretty(sess.target)
        size = ui.fit_title_size(name, 560, start=60, minimum=30)
        t = ui.title(name, size)
        ty = M + 54
        t.draw(canvas, M, ty)
        ty += t.height + 6
        p = ui.pill(f"Trial {min(sess.trial + (0 if sess.finished else 1), sess.trials)} / {sess.trials}", 15,
                    height=34)
        p.draw(canvas, M, ty)
        acc = sess.running_accuracy
        ui.pill(f"Accuracy {acc:.0%}" if acc is not None else "Accuracy —", 15, height=34).draw(canvas,
                                                                                             M + p.width + 10, ty)
        ui.progress_bar(300, 6, sess.trial / max(1, sess.trials)).draw(canvas, M + 2, ty + 44)

    # --- top-right: the category's signs (click = make it the target) --------------------------------
    states = sess.sign_states()
    names = list(states)
    room = max(1, (ui.UI_H - 260 - M) // 36)
    if len(names) > room:
        cur = names.index(sess.target) if sess.target in names else 0
        start = max(0, min(cur - room // 2, len(names) - room))
        shown = names[start:start + room]
    else:
        start, shown = 0, names
    yy = M
    if start > 0:
        more = ui.chip(f"{start} more above", "more")
        more.draw(canvas, ui.UI_W - M - more.width, yy)
        yy += more.height + 6
    for sgn in shown:
        st = states[sgn] if test else ("current" if sgn == sess.target else "next")
        c = ui.chip(pretty(sgn), st)
        cx = ui.UI_W - M - c.width
        c.draw(canvas, cx, yy)
        hit.add(cx, yy, c.width, c.height, f"sign:{sgn}")
        yy += c.height + 6
    rest = len(names) - start - len(shown)
    if rest > 0:
        more = ui.chip(f"{rest} more below", "more")
        more.draw(canvas, ui.UI_W - M - more.width, yy)

    # --- bottom-right: result card -----------------------------------------------------------------
    a = sess.last
    if a is not None:
        w, row = 340, 34
        top3 = a.top3[:3]
        h = 58 + row * len(top3) + 34
        colour = TONE[a.tone]
        heading = "Not sure, try again" if a.tone == "unsure" else pretty(a.prediction)
        texts = [(20, 30, heading, 20, "semibold", colour, 1.0, "lm")]
        for i, (lab, p) in enumerate(top3):
            yv = 58 + i * row + 12
            texts.append((20, yv, pretty(lab), 14, "regular", ui.WHITE, 0.9, "lm"))
            texts.append((w - 20, yv, f"{p:.0%}", 14, "semibold", ui.WHITE, 0.9, "rm"))
        texts.append((20, h - 18, f"Response {a.response_ms:.0f} ms" + (f"  ·  hand {a.hand_used}" if sess.hand ==
                                                                       "auto" else ""), 13, "regular", ui.MUTED,
                      1.0, "lm"))
        bx, by = ui.UI_W - M - w, ui.UI_H - M - h
        ui.box(w, h, 18, ui.DARK, 0.66, outline=colour, outline_alpha=0.9, outline_w=2,
               texts=tuple(texts)).draw(canvas, bx, by)
        for i, (lab, p) in enumerate(top3):
            ui.progress_bar(w - 40, 5, p, (colour, colour) if i == 0 else (ui.MUTED, ui.MUTED)).draw(
                canvas, bx + 20, by + 58 + i * row + 24)

    # --- bottom-left: buttons and the last 10 attempts ------------------------------------------------
    by = ui.UI_H - M - 36
    x = M
    for label, key, letter in (("Pause" if not sess.paused else "Start", "pause", "Space"), ("Next sign", "next", "N"),
                               ("Hand", "hand", "H"), ("Mode", "mode", "M"), ("Quit", "quit", "Q")):
        x = _with_badge(canvas, hit, ui.pill(label, 14, "medium", height=36, pad=14), x, by, key, letter) + 14
    hy = by - 46
    if sess.history:
        lab = ui.pill("Last 10", 13, "regular", ui.MUTED, height=30, pad=12)
        lab.draw(canvas, M, hy)
        hx = M + lab.width + 8
        for att in sess.history:
            icon = "check" if att.correct else "cross" if att.correct is False else "dot"
            col = ui.SUCCESS if att.correct else ui.DANGER if att.correct is False else TONE[att.tone]
            s = ui.round_button(icon, 30, bg=col, bg_alpha=0.85)
            s.draw(canvas, hx, hy)
            hx += s.width + 4

    # --- bottom centre: what to do now (+ the 3-second limit while signing) ---------------------------------
    cap = ui.pill(info.get("message") or sess.status_caption(), 18, "medium", height=46, pad=24, bg_alpha=0.66,
                  dot=ui.PRIMARY if sess.signing else None)
    cy = by - 46 - 12 - cap.height if sess.history else by - 12 - cap.height
    cap.draw(canvas, (ui.UI_W - cap.width) // 2, cy)
    if sess.signing:
        ui.progress_bar(240, 5, sess.window_fraction).draw(canvas, (ui.UI_W - 240) // 2, cy - 12)

    if sess.finished:
        draw_summary(canvas, sess, hit)
    return canvas, hit


def draw_summary(canvas, sess: LiveSession, hit: ui.HitMap) -> None:
    ui.dim(canvas, 0.55)
    s = sess.summary()
    rows = [("Accuracy", f"{s['accuracy']:.0%}" if s["accuracy"] is not None else "—", ui.WHITE, True),
            ("Correct", f"{s['correct']} of {s['attempts']}", ui.WHITE, False),
            ("Mean response time", f"{s['mean_response_ms']:.0f} ms" if s["mean_response_ms"] is not None else "—",
             ui.SUCCESS if (s["mean_response_ms"] or 9e9) <= 1000 else ui.WARNING, False)]
    for sign, p in list(s["per_sign"].items())[:8]:
        rows.append((pretty(sign), f"{p['correct']}/{p['attempts']}  ·  {p['mean_response_ms']:.0f} ms",
                     ui.SUCCESS if p["accuracy"] >= 0.85 else ui.WARNING, False))
    c = ui.card(rows, "Test finished", width=520, tag="")
    x, y = (ui.UI_W - c.width) // 2, max(M, (ui.UI_H - c.height) // 2 - 30)
    c.draw(canvas, x, y)
    again = ui.pill("New run", 15, "semibold", bg=ui.PRIMARY, bg_alpha=0.92, height=40, pad=18)
    xx = (ui.UI_W - again.width - 120) // 2
    _with_badge(canvas, hit, again, xx, y + c.height + 16, "pause", "Space")
    close = ui.pill("Close", 15, "medium", height=40, pad=18)
    _with_badge(canvas, hit, close, xx + again.width + 16, y + c.height + 16, "quit", "Q")


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


KEYS = {ord(" "): "pause", ord("n"): "next", ord("N"): "next", ord("h"): "hand", ord("H"): "hand", ord("m"): "mode",
        ord("M"): "mode", ord("q"): "quit", ord("Q"): "quit", 27: "quit"}


def run(ctx, category: str, mode: str = "free", tester: str = "", tester_hand: str = "right", target: str = "",
        trials: int = 25, hand: str | None = None, camera: int | None = None, backend: str | None = None,
        video: str | None = None, headless: bool = False, threshold: float | None = None,
        screenshot: str | None = None, recognizer: str = "trainer", capture: str = "fixed",
        export: str | None = None) -> dict:
    import cv2
    s = cfg.load_settings()
    paths = cfg.Paths(s)
    mdir = paths.model_dir(category)
    if recognizer == "app":
        from .app_export import export_dir
        edir = export_dir(paths, category, export)
    elif not (mdir / "model.keras").exists():
        raise ds.PlainError(f"No final model for '{category}'.", "Train it first on the Train model page.")
    thr = float(threshold if threshold is not None else s.get("confidence_threshold", 0.85))
    rec = recorder.recorder_camera_settings()
    camera = int(camera if camera is not None else (s.get("camera") if s.get("camera") is not None else rec["camera"]))
    backend = backend or s.get("backend") or rec.get("backend", "msmf")
    ctx.progress(0.0, "Loading the model...")
    if recognizer == "app":
        clf = AppClassifier(edir, capture)
        thr = clf.threshold
        engine = (f"App model v{clf.feature_version} {clf.version[:8]} · "
                  f"{'fixed capture' if capture == 'fixed' else 'app capture (2 s)'}")
    else:
        clf = Classifier(mdir, thr)
        capture = "fixed"
        engine = f"Trainer model v{clf.feature_version}"
    if mode == "test" and target and target not in clf.labels:
        raise ds.PlainError(f"'{target}' is not a sign of the {category} model.", "Pick a sign from the list.")
    sess = LiveSession(clf.labels, mode=mode, target=target or None, trials=int(trials),
                       hand=hand or s.get("default_hand") or "auto", threshold=thr)
    cap, fps, from_file = open_source(video, camera, backend)
    if not cap.isOpened():
        raise ds.PlainError(f"Could not open camera {camera} with {backend.upper()}.",
                            "Close Zoom/Teams/the Camera app, or pick another camera or backend in Settings.")
    shots = Path(screenshot) if screenshot else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    landmarker = create_landmarker(params=clf.params)
    # Feature v3 measures the hands against the face (README 6.7); the other versions don't need the face model.
    face_landmarker = create_face_landmarker() if int(clf.feature_version) == 3 else None
    seg = AppClipSegmenter(fps) if capture == "app" else LiveSegmenter(fps)
    t_hand: float | None = None                               # when the current window's first hand was seen
    from .features import frame_vector_rich
    clicked: list[str] = []
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, ui.UI_W, ui.UI_H)
        try:                                            # open in front of the browser
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except (cv2.error, AttributeError):
            pass
        hits = {"map": ui.HitMap()}

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                key = hits["map"].key_at(x, y)
                if key:
                    clicked.append(key)
        cv2.setMouseCallback(WINDOW, on_mouse)
    ctx.progress(0.0, "Live test window is open")
    ctx.log(f"Live test: {category}, mode {mode}, hand {sess.hand}, {engine} ({CAPTURES[capture]}), model "
            f"{clf.model_run_id} (canonical {'on' if clf.canonical else 'off'}), source "
            f"{'video ' + video if from_file else f'camera {camera} ({backend})'}")
    t_start = time.perf_counter()
    frame_i, last_ts = 0, -1
    fps_times: list[float] = []
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if from_file else 0
    canvas = None

    def act(key: str) -> bool:
        """A key or click; returns True to quit."""
        if key == "quit":
            return True
        if key == "pause":
            if sess.finished:
                sess.restart()
            else:
                sess.toggle_pause()
        elif key == "next":
            sess.next_sign()
        elif key == "hand":
            ctx.log(f"Hand setting: {sess.cycle_hand()}")
        elif key == "mode":
            sess.toggle_mode()
        elif key.startswith("sign:"):
            sess.pick(key[5:])
        return False

    def snap(name: str) -> None:
        if shots and name not in taken and canvas is not None:
            cv2.imwrite(str(shots / f"{name}.png"), canvas)
            taken.add(name)
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
                fps_times = (fps_times + [now])[-30:]
                if len(fps_times) >= 10:
                    seg.set_fps((len(fps_times) - 1) / (fps_times[-1] - fps_times[0]))
            ts = max(ts, last_ts + 1)
            last_ts = ts
            frame_i += 1
            hands = detect(landmarker, frame, ts)                      # UNMIRRORED frame
            face = detect_face(face_landmarker, frame, ts) if face_landmarker is not None else None
            h, w = frame.shape[:2]
            vec, _ = frame_vector_rich(hands, w, h, face)
            was_signing = seg.signing
            win = seg.push(vec, len(hands))
            if not was_signing and (seg.signing or win is not None):          # a window started with this frame
                t_hand = now
            if win is None and from_file and n_total and frame_i >= n_total:
                win = seg.flush()
            sess.signing = seg.signing
            sess.waiting_for_rest = seg.waiting_for_rest
            sess.window_fraction = (len(seg.active) / seg.n_max) if seg.active else 0.0
            attempt = None
            if win is not None and not sess.paused and not sess.finished:
                t_end = time.perf_counter()                            # end of the sign window
                r = clf.classify(win, sess.hand)
                r["response_ms"] = (time.perf_counter() - t_end) * 1000  # end of window -> prediction ready
                # hand in view -> prediction ready (the app team's "time to recognize"); not for a video file,
                # which runs faster than real time
                r["time_from_hand_ms"] = None if from_file or t_hand is None else (time.perf_counter() - t_hand) * 1000
                t_hand = None
                attempt = sess.record(r)
                ctx.log(f"attempt: {r['prediction']} {r['confidence']:.0%} "
                        f"({', '.join(f'{lab} {p:.0%}' for lab, p in r['top3'])}), hand {r['hand_used']}, "
                        f"{r['n_frames']} frames, {win.reason}, {r['response_ms']:.0f} ms after the sign window"
                        + (f", {r['time_from_hand_ms']:.0f} ms after the hand appeared" if r.get("time_from_hand_ms")
                           else ""))
                if attempt is not None and attempt.correct is not None:
                    save_attempt(paths, category, tester, tester_hand, attempt.target, attempt.prediction,
                                 attempt.confidence, attempt.correct, attempt.response_ms, clf.model_run_id,
                                 clf.feature_version, recognizer, capture, r.get("margin"), r.get("time_from_hand_ms"))
                    ctx.progress(sess.trial / max(1, sess.trials),
                                 f"Trial {sess.trial} / {sess.trials}: {sess.correct} correct")
            canvas, hitmap = draw_screen(frame, hands, sess, {"category": category, "hands_seen": bool(hands),
                                                              "engine": engine, "face": face})
            if attempt is not None:
                snap("summary" if sess.finished else sess.mode)
            if not headless:
                hits["map"] = hitmap
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                quit_ = False
                if key in KEYS:
                    quit_ = act(KEYS[key])
                while clicked:
                    quit_ = act(clicked.pop(0)) or quit_
                if quit_ or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            elif sess.finished:
                break
    finally:
        landmarker.close()
        if face_landmarker is not None:
            face_landmarker.close()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    msg = f"Live test finished: {ds.plural(len(sess.history) if mode == 'free' else sess.trial, 'attempt')}"
    if sess.trial:
        msg += f", {sess.correct} correct, {sess.correct / sess.trial:.0%}"
    done = sess.attempts if mode == "test" else list(sess.history)
    return {"message": msg + ".", "attempts": [dict(a.__dict__) for a in done], "correct": sess.correct,
            "screenshots": sorted(taken)}
