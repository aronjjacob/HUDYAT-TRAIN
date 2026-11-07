"""Clip playback (H.264 preview copies) and the animated 32-frame skeleton."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from trainer import settings as cfg
from trainer.recorder import HAND_CONNECTIONS


def h264_preview(src: Path) -> Path | None:
    """Browser-playable copy of a clip in work/preview_cache (the recorder saves mp4v, which browsers
    cannot play). The original is only read."""
    src = Path(src)
    if not src.exists():
        return None
    paths = cfg.Paths()
    st_ = src.stat()
    key = hashlib.sha1(f"{src.resolve()}|{st_.st_mtime_ns}|{st_.st_size}".encode()).hexdigest()[:20]
    out = paths.preview_cache / f"{key}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-an",
           "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run(cmd, capture_output=True, creationflags=flags, timeout=120)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, out)
    return out


def show_video(src: Path) -> None:
    """A clip (H.264 preview copy) or, for an uploaded photo, the photo itself."""
    from trainer.landmarks import IMAGE_EXT
    if not Path(src).exists():                       # e.g. a landmarks-only dataset: the samples, no videos
        st.html('<div class="note">No video in this dataset: landmarks only. The skeleton shows the sample.</div>')
        return
    if Path(src).suffix.lower() in IMAGE_EXT:
        st.image(str(src), width=340)
        return
    with st.spinner("Preparing the video..."):
        p = h264_preview(src)
    if p is None:
        st.html('<div class="note">This clip could not be converted for playback.</div>')
        return
    st.video(str(p), loop=True, autoplay=True, muted=True)


_SKELETON_HTML = """
<div id="wrap" style="font-family:Poppins,'Segoe UI',sans-serif;color:#1A1A1A">
  <canvas id="c" style="display:block;border:1px solid #E6E6E3;border-radius:8px;background:#fff;box-sizing:border-box"></canvas>
  <canvas id="tl" style="display:block;margin-top:6px"></canvas>
  <div id="cap" style="font-size:12px;color:#6B6B6B;margin-top:2px"></div>
</div>
<script>
const D = __DATA__;
const C = D.connections, ACC = "#1F4E99", AMB = "#B26A00", MUTED = "#6B6B6B", LINE = "#E6E6E3";
const c = document.getElementById("c"), x = c.getContext("2d");
const tl = document.getElementById("tl"), t = tl.getContext("2d");
const cap = document.getElementById("cap");
const W = Math.max(280, Math.min(640, document.getElementById("wrap").clientWidth || 520));
const H = Math.min(300, Math.round(W * 0.58));
const dpr = window.devicePixelRatio || 1;
function setup(cv, w, h) { cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + "px"; cv.style.height = h + "px";
                           cv.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0); }
setup(c, W, H); setup(tl, W, 46);
function hand(frame, off) { const p = []; for (let k = 0; k < 21; k++) p.push([frame[off+3*k], frame[off+3*k+1]]); return p; }
function drawHand(p, cx, cy, s, col) {
  x.strokeStyle = col; x.lineWidth = 2; x.lineCap = "round";
  for (const [a, b] of C) { x.beginPath(); x.moveTo(cx + p[a][0]*s, cy + p[a][1]*s); x.lineTo(cx + p[b][0]*s, cy + p[b][1]*s); x.stroke(); }
  for (const q of p) { x.beginPath(); x.arc(cx + q[0]*s, cy + q[1]*s, 3, 0, 7); x.fillStyle = "#fff"; x.fill(); x.lineWidth = 1.2; x.stroke(); }
}
function draw(j) {
  const f = D.sample[j];
  x.clearRect(0, 0, W, H);
  x.strokeStyle = LINE; x.lineWidth = 1; x.beginPath(); x.moveTo(W/2, 16); x.lineTo(W/2, H-16); x.stroke();
  x.font = "500 13px Poppins, sans-serif";
  const s = H / 3.4;
  [["Left", 0, 63, ACC, W*0.25], ["Right", 64, 127, AMB, W*0.75]].forEach(([name, off, flag, col, cx]) => {
    x.fillStyle = col; x.textAlign = "center"; x.fillText(name + " hand", cx, 24);
    if (f[flag] > 0.5) drawHand(hand(f, off), cx, H*0.80, s, col);
    else { x.fillStyle = "#B5B5B0"; x.fillText("not detected", cx, H/2); }
    x.textAlign = "left";
  });
  // timeline: whole clip, trim window, current position (not for a photo)
  t.clearRect(0, 0, W, 46);
  if (D.n_frames <= 1) { tl.style.display = "none"; cap.textContent = "Photo: a held-still sign, so the same hand is used for all 32 frames."; return; }
  const L = 4, R = W - 4, N = Math.max(1, D.n_frames - 1);
  const px = i => L + (R - L) * i / N;
  t.fillStyle = "#F7F7F5"; t.fillRect(L, 12, R - L, 12);
  t.fillStyle = "rgba(31,78,153,0.22)"; t.fillRect(px(D.trim_start), 12, Math.max(2, px(D.trim_end) - px(D.trim_start)), 12);
  const pos = D.trim_start + (D.trim_end - D.trim_start) * j / 31;
  t.fillStyle = ACC; t.fillRect(px(pos) - 1, 8, 2, 20);
  t.fillStyle = MUTED; t.font = "11px Poppins, sans-serif";
  t.fillText("0", L, 42); t.fillText(String(D.n_frames - 1), R - 18, 42);
  t.fillText("sign window " + D.trim_start + "–" + D.trim_end, Math.min(px(D.trim_start), R - 140), 42);
  cap.textContent = "Sample frame " + (j + 1) + " / 32  ·  clip frame " + Math.round(pos) + " of " + D.n_frames +
                    "  ·  shaded = sign window kept by the trim";
}
let j = 0;
const dt = Math.max(30, 1000 * D.window_s / 32);
draw(0);
setInterval(() => { j = (j + 1) % 32; draw(j); }, dt);
</script>
"""


def skeleton(sample: np.ndarray, n_frames: int, trim_start: int, trim_end: int, fps: float) -> None:
    data = {"sample": np.round(np.asarray(sample, dtype=float), 4).tolist(), "n_frames": int(n_frames),
            "trim_start": int(trim_start), "trim_end": int(trim_end),
            "window_s": max(0.5, (int(trim_end) - int(trim_start) + 1) / float(fps or 30)),
            "connections": [list(c) for c in HAND_CONNECTIONS]}
    components.html(_SKELETON_HTML.replace("__DATA__", json.dumps(data)), height=400)
