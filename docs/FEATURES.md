# Features, mirroring and the dominant hand

This extends **section 6 of the data collector's README** (the landmark contract). Everything in 6.1–6.6
still holds. This file says what the trainer stores per frame, how the model input is built from it,
and how left-handed users are handled.

The model input is exactly what the HUDYAT app gives its models (`client/src/features/fsl-recognition/normalize.ts`),
in one of two feature versions: **v1** `"hand-relative-v1"` (section 2) or **v2** `"hand-relative-sequence-v1"`
(section 2b, the format of Russel's trainer), so a model trained here gets the same numbers in the app. A third,
**v3** `"hand-face-location-v1"` (section 2c), adds where the hand is against the face; used for Family. The app
runs its face model only for v3 models (the export names it: `preprocessing.face`, catalog `faceUrl`). Every model has a `preprocess.json`; its `handedness` field says whether the frames are mirrored first. `tests/test_features.py` runs the JavaScript at the end of
this file and checks that it gives the same numbers as the trainer.

---

## 1. The stored frame: 149 values ("rich")

| Index | Content |
|---|---|
| 0 – 127 | The 128-value frame of README 6.2 / 6.3, unchanged |
| 128 – 130 | **Left wrist, raw**: landmark 0 as `(x · W/H, y, z · W/H)` (the 6.3 aspect correction, **before** the wrist is subtracted) |
| 131 – 133 | **Right wrist, raw** (same) |
| 134 | **Left hand size**: the 3D distance landmark 0 → landmark 9 in the same coordinates (the divisor of 6.3) |
| 135 | **Right hand size** |
| 136 – 141 | **Left hand location against the face** (README 6.7): wrist, thumb tip, index tip as `(x, y)` from the nose, in face sizes |
| 142 – 147 | **Right hand location** (same) |
| 148 | **Face present flag** (`1.0` / `0.0`) |

- A missing hand: its wrist and size are 0, like its 63 values and its flag, and its 6 location values are 0.
- No face found (MediaPipe FaceLandmarker, `numFaces: 1`, VIDEO mode, on the same unmirrored frame): all 13
  location values are 0. The math is exactly README 6.7 (nose = face landmark 1, forehead = 10, chin = 152).
- The hand in each slot is the same one 6.2 chose (by label; a shared label goes to the higher score).
- **Sign window (6.4)** is unchanged.
- **Resampling to 32 frames (6.5)** treats a hand's 4 extra values like its 63 values: they are
  interpolated when that hand is present in both neighbouring frames, otherwise copied from the
  nearest frame. A hand's 6 location values are interpolated only when that hand **and the face** are
  present in both neighbouring frames (README 6.5); the face flag is copied from the nearest frame.

Samples are stored **as recorded**. A left-handed signer's sample is not flipped when stored (README 6.1).

## 2. Model input (128 values per frame, the app's "hand-relative-v1")

Feature **v1**. Per present hand, every landmark minus the wrist, in pixels
(`x · W`, `y · H`, `z · W`), divided by the **2D** distance wrist → landmark 9 (x and y only). A hand whose
2D distance is below `1e-6` counts as absent (63 zeros, flag 0), as in the app.

The stored values 0–127 (README 6.3) are the same thing divided by the **3D** distance. So the model
input is computed from them: each present hand's 63 values divided by `hypot(x₉, y₉)` of its own stored
landmark 9. On a single frame this is exactly the app's value. On the 32 resampled frames it is the
3D-normalized frames resampled, then rescaled, which differs from the app's order (rescale, then
resample) only by the tiny change of the hand's tilt between two neighbouring frames.

The app stores the two hands at other indices (Left 0–62, Right 63–125, flags 126 and 127). The export
reorders the model's first layer for that (`trainer/app_export.py`); the trainer itself always uses the
layout above.

## 2b. Feature v2 ("hand-relative-sequence-v1", the wrist's movement)

v1 makes every frame relative to its own wrist, so the wrist is always 0: the model sees the hand's shape in
every frame but not where the hand travels (the stroke of a Z). v2 keeps that path:

- Landmarks 1–20 and the flags: exactly v1.
- Landmark 0 (the wrist), per hand: its raw position (values 128–130 / 131–133) minus its raw position in the
  hand's **first present frame of the clip**, divided by the **median 2D palm length** of the hand's present
  frames. The 2D palm length of a frame is `hypot(x₉, y₉) · size` (the stored values are divided by the 3D
  size, 134 / 135). Absent frames stay 0.
- A still hand (a photo, or the same frame repeated) has no movement: v2 gives exactly v1.

The trainer computes v2 on the **32 resampled frames** (the stored samples), like every other step of the
model input. Russel's trainer computes it on every camera frame and resamples after: landmarks 1–20 agree,
but the wrist path can start up to one 32-frame step later on the resampled frames (they may skip the frame
where the hand appeared). So a trainer v2 export says `"wrist_path": "after resampling …"` in its
`preprocessing`, and the app must convert after resampling for it (`tests/test_features.py` measures the
difference). Mirroring commutes with v2 (section 3), so the order of mirror and v2 does not matter.

## 2c. Feature v3 ("hand-face-location-v1", where the hand is)

v1 and v2 put every hand at the same place, so two signs with the same hand shape made at different places
(Father at the forehead, Mother at the chin) give the same numbers. v3 is **v1 (128 values) followed by the 13
location values 136–148** of the stored frame: **141 values**, the frame of README 6.2. A hand that v1 counts as
absent gets 6 location zeros.

## 3. Mirroring (a left-handed version of a sample)

On any layout (128, 141 or 149 values per frame):

1. Negate every **x**: the 21 x values of each hand (indices `0, 3, …, 60` and `64, 67, …, 124`), and
   the x of the raw wrists / movements (128 and 131), and the x of the location points (149 layout:
   136, 138, 140, 142, 144, 146; v3 input: 128, 130, 132, 134, 136, 138).
2. **Swap the Left and Right parts**: values 0–63 ↔ 64–127 (the flags move with their hand), in the
   149 layout 128–130 ↔ 131–133, 134 ↔ 135 and 136–141 ↔ 142–147, in the v3 input 128–133 ↔ 134–139.
   The face flag stays.

Mirroring twice gives back the original. The raw wrist positions are only ever used as differences, so
negating x is the mirror image; the location points are measured from the nose, so negating their x is too.

## 4. The dominant hand (canonical form)

The models are trained on **right-dominant** signing. A left-handed person signs the mirror image, so
before the model sees a sign:

```
if the user's hand is "left": frames = mirror(frames)        // on the 32 x 149 window, before the model input
```

The hand comes from the app's **Hand** setting: Right, Left or **Auto**. Auto decides per sign window
(the 32 resampled 149-value frames):

1. No hand in any frame → right.
2. If more than half of the frames that have a hand have **exactly one** hand, the dominant hand is the
   one seen alone more often: labelled Left → left, otherwise right.
3. Otherwise (mostly two hands), the hand whose wrist **travels further** is dominant. Path length =
   sum of the distances between consecutive present raw wrist positions, divided by that hand's median
   size. Left travels further → left; a tie → right.

`preprocess.json` → `"handedness": {"canonical": true}` means the model expects this step.
`"canonical": false` means the frames go in as recorded.

## 5. Order of the steps in the browser app

```
per video frame:  MediaPipe hands + face (6.1, 6.7)  →  149-value frame (section 1)
per sign window:  6.6 window  →  6.4 trim  →  resample to 32 (section 1 rule)
                  →  if canonical: hand = setting or autoHand(window); if hand is left: mirror
                  →  modelInput(window)  (section 2; v2: modelInputSequence(window), section 2b;
                                          v3: modelInputLocation(window), section 2c)
                  →  model
```

<!-- js-reference -->
## 6. JavaScript reference

```js
// Landmarks may be MediaPipe objects {x, y, z} or arrays [x, y, z].
const XYZ = q => Array.isArray(q) ? q : [q.x, q.y, q.z];

function normalizeHand(lm, W, H) {                 // README 6.3
  const a = W / H;
  const p = lm.map(q => { const [x, y, z] = XYZ(q); return [x * a, y, z * a]; });
  const [wx, wy, wz] = p[0];
  for (const q of p) { q[0] -= wx; q[1] -= wy; q[2] -= wz; }
  const s = Math.hypot(p[9][0], p[9][1], p[9][2]);
  const out = p.flat();
  return s > 1e-6 ? out.map(v => v / s) : out;
}

// hands = [{label: "Left"|"Right", score, landmarks: 21 points}] as MediaPipe returns them (unmirrored frame);
// face = the FaceLandmarker's 478 points, or null when no face was found.
function frameVectorRich(hands, W, H, face = null) {
  const best = {};
  for (const h of hands) {
    if (h.label !== "Left" && h.label !== "Right") continue;
    if (!best[h.label] || h.score > best[h.label].score) best[h.label] = h;
  }
  const v = new Array(149).fill(0);
  const a = W / H;
  [["Left", 0, 63, 128, 134], ["Right", 64, 127, 131, 135]].forEach(([label, off, flag, wr, sc]) => {
    const h = best[label];
    if (!h) return;
    normalizeHand(h.landmarks, W, H).forEach((x, i) => { v[off + i] = x; });
    v[flag] = 1;
    const p0 = XYZ(h.landmarks[0]), p9 = XYZ(h.landmarks[9]);
    v[wr] = p0[0] * a; v[wr + 1] = p0[1]; v[wr + 2] = p0[2] * a;
    v[sc] = Math.hypot((p9[0] - p0[0]) * a, p9[1] - p0[1], (p9[2] - p0[2]) * a);
  });
  if (!face) return v;                            // README 6.7: location against the face
  const P = i => { const [x, y] = XYZ(face[i]); return [x * a, y]; };
  const [N, F, C] = [P(1), P(10), P(152)];
  const s = Math.hypot(F[0] - C[0], F[1] - C[1]);
  if (s < 1e-6) return v;
  ["Left", "Right"].forEach((label, h) => {
    if (!best[label]) return;
    [0, 4, 8].forEach((k, j) => {
      const [x, y] = XYZ(best[label].landmarks[k]);
      v[136 + h * 6 + j * 2] = (x * a - N[0]) / s;
      v[136 + h * 6 + j * 2 + 1] = (y - N[1]) / s;
    });
  });
  v[148] = 1;
  return v;
}

const range = (a, b) => Array.from({length: b - a}, (_, i) => a + i);
function slots(width) {                           // [value indices, flags that must be present, flag to copy]
  const left = range(0, 63), right = range(64, 127);
  if (width !== 149) return [[left, [63], 63], [right, [127], 127]];
  left.push(128, 129, 130, 134); right.push(131, 132, 133, 135);
  return [[left, [63], 63], [right, [127], 127],
          [range(136, 142), [63, 148], null], [range(142, 148), [127, 148], null], [[], [], 148]];
}

function resampleRich(frames, n = 32) {           // README 6.5, with each hand's extra values
  const T = frames.length, D = frames[0].length, S = slots(D);
  const out = [];
  for (let j = 0; j < n; j++) {
    const t = n > 1 ? j * (T - 1) / (n - 1) : 0;
    const i0 = Math.floor(t), i1 = Math.min(i0 + 1, T - 1), w = t - i0;
    const near = w < 0.5 ? i0 : i1;
    const o = new Array(D).fill(0);
    for (const [idx, flags, flag] of S) {
      const both = flags.every(f => frames[i0][f] > 0.5 && frames[i1][f] > 0.5);
      for (const k of idx) o[k] = both ? (1 - w) * frames[i0][k] + w * frames[i1][k] : frames[near][k];
      if (flag !== null) o[flag] = frames[near][flag];
    }
    out.push(o);
  }
  return out;
}

const median = xs => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
                       return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

function modelInput(win) {                        // win = 32 frames of 149 values -> 32 frames of 128 (section 2)
  return win.map(f => {
    const o = f.slice(0, 128);
    [[0, 63], [64, 127]].forEach(([off, flag]) => {
      const k = Math.hypot(o[off + 27], o[off + 28]);           // landmark 9's x and y
      if (o[flag] > 0.5 && k > 1e-6) { for (let i = off; i < off + 63; i++) o[i] /= k; return; }
      for (let i = off; i < off + 63; i++) o[i] = 0;
      o[flag] = 0;
    });
    return o;
  });
}

function modelInputSequence(win) {                // feature v2 (section 2b): modelInput + each wrist's movement
  const out = modelInput(win);
  [[0, 63, 128, 134], [64, 127, 131, 135]].forEach(([off, flag, wr, sc]) => {
    const valid = out.map((f, t) => f[flag] > 0.5 ? t : -1).filter(t => t >= 0);
    if (!valid.length) return;
    const s = median(valid.map(t => Math.hypot(win[t][off + 27], win[t][off + 28]) * win[t][sc]));
    const o = valid[0];
    for (const t of valid) for (let c = 0; c < 3; c++) out[t][off + c] = (win[t][wr + c] - win[o][wr + c]) / s;
  });
  return out;
}

function modelInputLocation(win) {                // feature v3 (section 2c): modelInput + the 13 location values
  return modelInput(win).map((o, t) => {
    const loc = win[t].slice(136, 149);
    for (let i = 0; i < 6; i++) { if (o[63] < 0.5) loc[i] = 0; if (o[127] < 0.5) loc[6 + i] = 0; }
    return o.concat(loc);
  });
}

function mirrorFeatures(frames) {                 // works on 128, 141 or 149 values per frame
  return frames.map(f => {
    const D = f.length, x = f.slice();
    for (let i = 0; i < 63; i += 3) { x[i] = -x[i]; x[64 + i] = -x[64 + i]; }
    const o = x.slice(64, 128).concat(x.slice(0, 64));
    if (D === 149) {
      for (const i of [128, 131, 136, 138, 140, 142, 144, 146]) x[i] = -x[i];
      o.push(...x.slice(131, 134), ...x.slice(128, 131), x[135], x[134],
             ...x.slice(142, 148), ...x.slice(136, 142), x[148]);
    } else if (D === 141) {
      for (const i of [128, 130, 132, 134, 136, 138]) x[i] = -x[i];
      o.push(...x.slice(134, 140), ...x.slice(128, 134), x[140]);
    }
    return o;
  });
}

function autoHand(win) {                          // win = 32 frames of 149 values -> "right" | "left"
  const lp = win.map(f => f[63] > 0.5), rp = win.map(f => f[127] > 0.5);
  const anyH = lp.filter((l, i) => l || rp[i]).length;
  if (anyH === 0) return "right";
  const onlyL = lp.filter((l, i) => l && !rp[i]).length, onlyR = rp.filter((r, i) => r && !lp[i]).length;
  if ((onlyL + onlyR) * 2 > anyH) return onlyL > onlyR ? "left" : "right";
  const path = (pres, wr, sc) => {
    const pts = win.filter((f, i) => pres[i]);
    if (pts.length < 2) return 0;
    let d = 0;
    for (let i = 1; i < pts.length; i++)
      d += Math.hypot(pts[i][wr] - pts[i - 1][wr], pts[i][wr + 1] - pts[i - 1][wr + 1], pts[i][wr + 2] - pts[i - 1][wr + 2]);
    const s = median(pts.map(f => f[sc]));
    return d / (s > 1e-6 ? s : 1);
  };
  return path(lp, 128, 134) > path(rp, 131, 135) ? "left" : "right";
}
```
