# Recommended recognition for moving signs in the HUDYAT app

For the HUDYAT app team (`client/src/features/fsl-recognition/`). Nothing here changes the model, its files, the
hand detector or `normalizeFrame`. It changes **which frames** reach the model and **how they are squeezed to 32**,
so the model gets clips like the ones it was trained on. The trainer's Live test runs exactly this
(*Recognizer: HUDYAT app model*, *Capture: Fixed*) next to what the app does today (*Capture: As the app today*),
so both can be tried with the same tester and signs before anything changes in the app.

## What changes, and why

| | The app today (`category-recognition.tsx`, `category-core.js`) | Recommended | Why |
|---|---|---|---|
| **When a sign ends** | A fixed **2-second** clip from the first frame with a hand | When **no hand has been seen for 0.2 s** (or after 3 s at most) | The models are trained on clips cut to the sign (first to last frame with a hand, ±0.1 s). A 2 s clip of a 1.4 s sign gives the model ~0.6 s of hands going down, which it never saw in training. It also makes every answer wait the full 2 s. |
| **Cutting the clip** | Not cut | Cut to the frames with a hand, plus 0.1 s on each side, at least 8 frames (recorder README 6.4) | Same as training. |
| **Squeezing to 32 frames** | Every coordinate interpolated between two frames even when the hand is missing in one of them (so values shrink towards 0), then zeroed if the nearest frame has no hand | Interpolate a hand only when it is present in **both** frames; otherwise copy the nearest frame (recorder README 6.5) | Same as training; no half-size hands. |
| **Left-handed learners** | Not handled | Mirror the clip to the right-handed form before the model (hand: Right / Left / Auto) | The models are trained on right-handed signing (all signers are right-handed). |
| **Answer** | Same model, same gates (≥ 80 % and ≥ 15 points ahead) | Unchanged | |

**Response time.** With the recommended capture, the answer comes about 0.2 s after the hands go down (plus
the model's time), not 2 s after they came up. The trainer's Live test records both measures for every
attempt: *after the sign ends* and *after the hand appears* (the app's "time to recognize").

Letters (the alphabet's hold rule) are not affected.

**Feature v2 models** (`preprocessing.normalization: "hand-relative-sequence-v1"` with
`preprocessing.wrist_path: "after resampling …"`, docs/FEATURES.md section 2b): cut and resample the raw
frames with each hand's raw wrist and palm length travelling with it (`resampleSign`'s `extra`), then convert
the 32 frames (`modelInputSequence`), then mirror a left-handed sign. A v2 model without `wrist_path` (Russel's
trainer) converts every frame first (the app's `normalizeSequence`) and resamples after.

**Feature v3 models** (`preprocessing.normalization: "hand-face-location-v1"`, Family; docs/FEATURES.md
section 2c): the app also runs MediaPipe FaceLandmarker (`preprocessing.face`, catalog `faceUrl`) on the same
frame and timestamp, and each item's `norm` is `normalizeFrameLocation` (141 values: `normalizeFrame` plus
where the wrist, thumb tip and index tip are against the face). `resampleSign` and `mirrorFrames` below handle
the 13 extra values; `tests/test_live_app.py` checks the result against the trainer.

## The order of the steps

```
every frame:   frame = packResult(detector result)            // unchanged
               segmenter.push({raw: frame, norm: normalizeFrame(frame, W, H)}, hasHand(frame))
a sign ended:  {input, hand} = recommendedInput(items, fps, W, H, handSetting)
               model.predict(input)                            // unchanged; same gates
```

`fps` is the capture rate the frames actually arrived at (frames / seconds of the window); the app polls about
every 40 ms plus processing time.

<!-- app-js -->
## JavaScript (the app's frame layout: Left 0–62, Right 63–125, flags 126 and 127)

```js
const PRE_BUFFER_S = 0.1, END_ABSENT_S = 0.2, MAX_WINDOW_S = 3.0, MIN_HANDS_S = 0.2;
const roundHalfUp = x => Math.floor(x + 0.5);
const hasHand = f => f[126] > 0.5 || f[127] > 0.5;

// A sign starts with the first frame with a hand (with the 0.1 s before it) and ends when no hand has been seen for
// 0.2 s, or after 3 s. After a 3-second window the hands must go down before the next sign. A window with less than
// 0.2 s of hands is a detection blip: no sign. push() returns {items, reason} when a sign has ended, else null.
function createSignSegmenter(fps) {
  const n = s => Math.max(1, roundHalfUp(s * fps));
  const nPre = n(PRE_BUFFER_S), nEnd = n(END_ABSENT_S), nMax = n(MAX_WINDOW_S), nMinHands = n(MIN_HANDS_S);
  let ring = [], active = null, absent = 0, waitingForRest = false;
  const remember = entry => { ring.push(entry); if (ring.length > nPre) ring.shift(); };
  const finish = reason => {
    const items = active;
    active = null; absent = 0; waitingForRest = reason === "max_length";
    if (items.filter(e => e.present).length < nMinHands) return null;
    return {items: items.map(e => e.item), reason};
  };
  return {
    get signing() { return active !== null; },
    push(item, present) {
      const entry = {item, present};
      if (!active && waitingForRest) {
        absent = present ? 0 : absent + 1;
        if (absent >= nEnd) { waitingForRest = false; absent = 0; }
        remember(entry);
        return null;
      }
      if (!active) {
        if (!present) { remember(entry); return null; }
        active = [...ring, entry]; ring = []; absent = 0;
        return active.length >= nMax ? finish("max_length") : null;
      }
      active.push(entry);
      absent = present ? 0 : absent + 1;
      if (absent >= nEnd) return finish("hands_gone");
      return active.length >= nMax ? finish("max_length") : null;
    },
  };
}

// Recorder README 6.4: the frames with a hand, plus 0.1 s on each side, at least 8 frames. Inclusive [start, end].
function signWindow(present, fps) {
  const T = present.length;
  const first = present.indexOf(true), last = present.lastIndexOf(true);
  if (first < 0) return [0, T - 1];
  const m = roundHalfUp(0.1 * fps);
  let start = Math.max(0, first - m), end = Math.min(T - 1, last + m);
  const need = Math.min(8, T);
  if (end - start + 1 < need) {
    const centre = (start + end) / 2;
    start = Math.max(0, Math.min(roundHalfUp(centre - (need - 1) / 2), T - need));
    end = start + need - 1;
  }
  return [start, end];
}

// Recorder README 6.5 in the app's layout: a hand is interpolated only when it is present in both frames,
// otherwise copied from the nearest frame; its flag always comes from the nearest frame.
// `extra` (optional) = per hand, more values that travel with it: [[indices of hand 0], [indices of hand 1]].
// Feature v3 frames (141 values): a hand's location against the face (128-133 Left, 134-139 Right) is interpolated
// only when the hand and the face (flag 140) are in both frames; the face flag comes from the nearest frame.
function resampleSign(frames, n = 32, extra = null) {
  const T = frames.length, D = frames[0].length, out = [];
  const slots = [0, 1].map(h => {
    const idx = Array.from({length: 63}, (_, i) => h * 63 + i);
    return [extra ? idx.concat(extra[h]) : idx, [126 + h], 126 + h];
  });
  if (D === 141) slots.push([[128, 129, 130, 131, 132, 133], [126, 140], null],
                            [[134, 135, 136, 137, 138, 139], [127, 140], null], [[], [], 140]);
  for (let j = 0; j < n; j++) {
    const t = n > 1 ? j * (T - 1) / (n - 1) : 0;
    const i0 = Math.floor(t), i1 = Math.min(i0 + 1, T - 1), w = t - i0;
    const near = w < 0.5 ? i0 : i1;
    const o = new Array(D).fill(0);
    for (const [idx, flags, flag] of slots) {
      const both = flags.every(f => frames[i0][f] > 0.5 && frames[i1][f] > 0.5);
      for (const k of idx) o[k] = both ? (1 - w) * frames[i0][k] + w * frames[i1][k] : frames[near][k];
      if (flag !== null) o[flag] = frames[near][flag];
    }
    out.push(o);
  }
  return out;
}

// The mirror image of normalized frames (a left-handed signer -> right-handed): every x negated, hands swapped.
// Feature v3 (141 values): the location x negated too, Left and Right location swapped, the face flag stays.
function mirrorFrames(frames) {
  return frames.map(f => {
    const x = Array.from(f);                          // plain array: frames may be Float32Array
    for (let i = 0; i < 126; i += 3) x[i] = -x[i];
    const out = x.slice(63, 126).concat(x.slice(0, 63), [x[127], x[126]]);
    if (x.length !== 141) return out;
    for (let i = 128; i < 140; i += 2) x[i] = -x[i];
    return out.concat(x.slice(134, 140), x.slice(128, 134), [x[140]]);
  });
}

// Which hand signs (trainer docs/FEATURES.md section 4), from the RAW frames of the cut window (packResult output).
// Mostly one hand visible -> that hand; mostly two -> the hand whose wrist travels further (in hand sizes).
function autoHand(raws, W, H) {
  const a = W / H;
  const rich = raws.map(f => {                        // per hand: wrist (x*a, y, z*a), size |landmark 9 - wrist|
    const o = Array.from(f);                          // plain array (it grows to 136); frames may be Float32Array
    o.length = 136;
    [0, 1].forEach(h => {
      const b = h * 63, base = 128 + 4 * h;
      if (!(f[126 + h] > 0.5)) { o[base] = o[base + 1] = o[base + 2] = o[base + 3] = 0; return; }
      o[base] = f[b] * a; o[base + 1] = f[b + 1]; o[base + 2] = f[b + 2] * a;
      o[base + 3] = Math.hypot((f[b + 27] - f[b]) * a, f[b + 28] - f[b + 1], (f[b + 29] - f[b + 2]) * a);
    });
    return o;
  });
  const win = resampleSign(rich, 32, [[128, 129, 130, 131], [132, 133, 134, 135]]);
  const lp = win.map(f => f[126] > 0.5), rp = win.map(f => f[127] > 0.5);
  const anyH = lp.filter((l, i) => l || rp[i]).length;
  if (anyH === 0) return "right";
  const onlyL = lp.filter((l, i) => l && !rp[i]).length, onlyR = rp.filter((r, i) => r && !lp[i]).length;
  if ((onlyL + onlyR) * 2 > anyH) return onlyL > onlyR ? "left" : "right";
  const median = xs => { const s = [...xs].sort((p, q) => p - q), m = s.length >> 1;
                         return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const path = (pres, base) => {
    const pts = win.filter((f, i) => pres[i]);
    if (pts.length < 2) return 0;
    let d = 0;
    for (let i = 1; i < pts.length; i++)
      d += Math.hypot(pts[i][base] - pts[i - 1][base], pts[i][base + 1] - pts[i - 1][base + 1],
                      pts[i][base + 2] - pts[i - 1][base + 2]);
    const s = median(pts.map(f => f[base + 3]));
    return d / (s > 1e-6 ? s : 1);
  };
  return path(lp, 128) > path(rp, 132) ? "left" : "right";
}

// items: what the segmenter returned, each {raw: packResult frame, norm: normalizeFrame(raw, W, H)} (feature v3:
// norm = normalizeFrameLocation, 141 values). hand: "right" | "left" | "auto" (the learner's setting).
// Returns the model's (32 x 128, or 32 x 141) input and the hand used.
function recommendedInput(items, fps, W, H, hand = "auto") {
  const [s, e] = signWindow(items.map(it => hasHand(it.raw)), fps);
  const cut = items.slice(s, e + 1);
  let input = resampleSign(cut.map(it => it.norm));
  const used = hand === "auto" ? autoHand(cut.map(it => it.raw), W, H) : hand;
  if (used === "left") input = mirrorFrames(input);
  return {input, hand: used};
}
```

`tests/test_live_app.py` runs this JavaScript in Node and checks every function against the trainer's Live test,
including the whole path from raw frames to the model's input.
