# HUDYAT Trainer

Preprocesses and trains on the FSL clips recorded with the data-collection tool
(**HUDYAT - HAND TRACKER**). One model per category, evaluated on signers the model
never saw, exported to TensorFlow.js for the browser app, and tried live with a webcam.

The data-collection folder is **only read**. Everything the trainer makes goes into
this folder's `work\` folder.

---

## 1. Setup (once per laptop)

1. Install **Python 3.12 (64-bit)** from python.org if it isn't installed yet
   (the same one the recorder uses).
2. Double-click **`SETUP (run once).bat`**. It creates the Python environment (`venv\`),
   installs the packages from `requirements.txt`, and copies and checks the MediaPipe hand
   model. The first run takes 5–15 minutes.
3. Double-click **`START HUDYAT TRAINER.bat`**. A black window opens (keep it open while
   you work) and the app opens in the browser at http://localhost:8517.
4. The app finds `Documents\HUDYAT - HAND TRACKER` by itself and uses its recordings.
   To use other recordings, open **Dataset → Change dataset** (or Settings) and either
   - **Browse...** to another folder and press **Use this folder**, or
   - **upload a .zip**: a zipped copy of a HAND TRACKER folder, or only its `dataset\raw` folder
     (videos in `category\sign\S1–S7\file.mp4`; the clip list is then made from the folder names).
     Uploads are unpacked into `work\uploaded_data\`.

The step list on the left is always shown. If an old browser tab hid it, click
**Show steps** at the top left.

**Desktop shortcut:** right-click `START HUDYAT TRAINER.bat` → *Show more options* →
*Send to* → *Desktop (create shortcut)*. Double-clicking the shortcut starts the app.
If the app is already running, it just opens the browser again.

To stop the app, close the black window.

---

## 2. The order of the steps

**Datasets:** the **Dataset** box at the top of the sidebar switches between your HAND TRACKER
recordings and any uploaded dataset. Each dataset keeps its own samples, splits, models and reports
(uploads: `work\imports\<name>\work`), so nothing gets mixed up. To add one: **1. Dataset → Add a
dataset**, then either upload a **.zip** or pick a **folder** (Browse...). Accepted:

- **one folder per sign** with photos and/or videos (e.g. `Collated\A\*.jpg`, `Collated\B\*.jpg`):
  one model, fixed split 70 % train / 15 % validation / 15 % test per sign (copies of the same photo,
  like Roboflow's `x_jpg.rf.<code>.jpg`, stay together). A photo becomes a held-still 32-frame sample;
  its hand is stored as a right hand (left-hand photos are mirrored).
- **videos per signer** `category\sign\S1…S7\file.mp4`: split by signer like the recordings.
- **a copy of a HUDYAT - HAND TRACKER folder**.

Then run the steps as usual: Preprocess → Split → Train → Evaluate → Export → Live Test.

**One category at a time:** at the top of the sidebar, **Working on** chooses a category
(e.g. `colors`). Every step then works on that category only: Preprocess extracts colors
(+ `none`, which every model needs), Split / Train / Evaluate / Export / Live Test use the
colors model, and the ticks in the sidebar show colors' progress. Choose **All categories**
to work on everything at once (each page then has its own picker).

The sidebar lists the 8 steps. Each one shows its state:
**✓ done**, **○ ready** (can run now) or **🔒 locked** (an earlier step is needed first).
"Next:" under the list says what to do next. Buttons that can't run yet are greyed out,
with the reason written under them.

| Step | Page | What it does | Run it again when... |
|---|---|---|---|
| 1 | **Dataset** | Clips per sign × signer (S1–S7) against the targets in `signs.csv`; counts of kept / needs re-record / rejected / low-detection clips. Read only. | (always up to date) |
| 2 | **Preprocess** | Choose **what to preprocess**: all categories, or one category at a time (also `none`, and the static A–Y test set `alphabet_static_test`, which is only extracted when you choose it). **Extract new clips**: MediaPipe on each kept clip → 32 × 128 landmark sample (README section 6 of the recorder). **Re-extract** redoes the chosen clips. **Adjust preprocessing**: minimum hand detection (safe), and under *Advanced* the trim margin, minimum window and MediaPipe confidences (these change the landmark format: the values are saved with every sample and model and exported as `preprocess.json` for the browser app; re-extract after changing them). Also: a table per category (to do / done / made with other settings), the trim summary, the low-detection clips, and a clip next to its animated 32-frame skeleton. | new clips were recorded |
| 3 | **Validation** (optional) | A quality check by a person: watch clips and **Reject** bad ones (wrong sign, hands out of frame...) with a reason and your name, or **Approve** good ones. Rejected clips are left out of training (run Split again afterwards). Skip it and every kept clip is used. | an expert reviews clips |
| 4 | **Split** | Train = S1–S4, validation = S5, test = S6–S7 (by signer, fixed). Every model gets the `none` class. Shows counts per sign and warns about missing signs. | after steps 2 or 3 changed something |
| 5 | **Train** | One model per category (or all), with a live loss/accuracy chart. Settings: epochs (150), batch (32), augmentation factor (5), seed (42). | you want a better model |
| 6 | **Evaluate** | **Once**, when tuning is finished (tick box): accuracy, macro precision/recall/F1 per test signer and averaged, recall per sign, confusion matrix, inference time, **PASS/FAIL**. | rarely (see 4.3) |
| 7 | **Export** | TensorFlow.js files for the browser app, checked against the Keras model. | after training |
| 8 | **Live Test** | Opens a camera window. **Free** mode shows the top-3 guesses; **Test** mode scores a tester on a target sign (default 25 trials) and saves every attempt. Shows live accuracy and response time per sign and category. | |

Only one long job (extract, split, train, evaluate, export, live test) runs at a time.
A job keeps running if you refresh the page or close the browser tab; reopen the page
to see its progress. **Stop** ends it (finished parts are kept).

Errors appear in the page in plain language with what to do next. The full log is
in the *Full log* box under the message (send it to the developers if needed).

---

## 3. Which clips are used

- Only our signers **S1–S7** (the imported public `PUB` clips are ignored).
- Only `status = kept`; not `expert_status = rejected` (if that column exists in
  `metadata.csv`); not rejected on the Validation page.
- Hand detection inside the sign window **≥ 80 %** (change it in Settings), except the
  `none` class.
- `alphabet_static_test` is not trained here (it is the test set of the static A–Y photo model).

## 4. How it works

### 4.1 Preprocessing (identical to the browser app)
Recorder README section 6: MediaPipe HandLandmarker (VIDEO mode, 2 hands, the same
`hand_landmarker.task`) on the **unmirrored** frames → 128 values per frame (Left 63 + flag,
Right 63 + flag, slots by handedness label) → per-hand normalization (aspect correction,
wrist at the origin, divided by the wrist–MCP9 distance) → sign window → 32 frames.
The trim rule, the handedness rule and the MediaPipe settings are **imported from the
recorder's own `hudyat_common.py`** (Settings → Checks shows this). If it cannot be
imported, the trainer's copies are used; a test checks they give identical results.

### 4.2 Training
Augmentation on the training set only, in memory, fixed seed, ×5: noise, scale ±10 %,
rotation ±15° about the wrist, time warp, frame dropout; mirroring only for signs with
`mirror_ok = 1`; no shift; an absent hand stays zero.
Model (input 32 × 128): Conv1D 64 → BN → MaxPool → Dropout → Conv1D 128 → BN → Dropout →
BiLSTM 128 → Dropout → BiLSTM 64 → Dropout → Dense 64 → softmax (signs + none).
Adam 1e-3, early stopping on validation loss (patience 20, best weights restored),
learning-rate reduction on plateau.

### 4.3 Evaluation
Test signers S6 and S7, no augmentation. PASS = accuracy and macro precision/recall/F1
≥ 85 % (average of S6 and S7) **and** every sign's recall ≥ 70 % (S6 + S7 together).
Use the test set **once**, after tuning: if you retrain because of test results, the
test result no longer tells you how the model does on new people.

### 4.4 Live test
The camera window finds each sign itself (README 6.6): it starts when a hand appears
(with the 0.1 s before it), and ends when no hand has been seen for 0.2 s or after 3 s.
The preview is mirrored like a selfie; MediaPipe runs on the unmirrored frames.
Below 70 % confidence it shows "Not sure, try again". In Test mode an attempt is correct
when the guess is the target sign with ≥ 70 % confidence. Response time = end of the
sign window → prediction shown (target: under 1 second).
The camera comes from the recorder's `settings.json` unless you choose another in Settings.

---

## 5. Where the outputs go

```
work\
├── landmarks\<category>\<sign_id>\<signer>\<file>.npy   (32, 128) float32 samples
├── landmarks_index.csv                                  one row per sample (trim, detection rates, status)
├── landmarks_duplicate_handedness.log                   frames where both hands got the same label
├── validation.csv                                       Approve / Reject decisions (last one counts)
├── splits\<category>.json                               file lists + counts per split
├── models\<category>\
│   ├── model.keras  labels.json  config.json  history.csv  training_plot.png
│   └── tfjs\  model.json  group1-shard1of1.bin  labels.json  verify.json
├── reports\<category>\  report.md  metrics.json  confusion_matrix.png / .csv (+ per signer)
├── reports\summary.csv                                  one row per evaluated category
├── live_tests\<category>.csv                            every Test-mode attempt
├── preview_cache\                                       browser-playable copies of clips (safe to delete)
└── jobs\                                                logs and status of the background jobs
```

`trainer_settings.json` (this folder) holds the data-collection folder, camera and thresholds.

---

## 6. Without the app (command line)

Every button runs one of these scripts, so everything also works from a terminal
(`venv\Scripts\activate` first):

```
python extract.py [--all] [--category greetings ...]       (alphabet_static_test only when named)
python split.py [--category colors ...]
python train.py --category colors|all [--epochs 150 --batch 32 --aug 5 --seed 42]
python evaluate.py --category colors|all --confirm
python export_tfjs.py --category colors|all
python live_test.py --category colors [--mode test --tester NAME --target red --trials 25]
```

## 7. Tests (for the developers)

```
venv\Scripts\python -m pytest -m "not slow"     unit tests (about 1 minute)
venv\Scripts\python -m pytest                   + end-to-end and UI tests (about 10 minutes)
```

- **Unit tests**: landmark format pinned to README section 6 (ports of its JavaScript
  reference code), sign-window parity with the recorder's `hudyat_common.py`, split rules
  (no signer in two splits, no PUB), augmentation (absent hand stays zero, mirror only when
  `mirror_ok`), the job lock, clip selection, the live sign window, and the TF.js export.
- **End-to-end** (`tests/test_end_to_end.py`): copies real FSL-105 clips from the recorder
  folder into a temporary folder, relabelled S1–S7, and runs extract → split → train →
  evaluate → export through the same functions the buttons use, checks every output file,
  compares TF.js with Keras, and feeds a recorded video through the live test (hidden
  `--video` option). Its numbers are only a pipeline check.
- **UI** (`tests/test_ui.py`): starts the app headless and clicks through every page with
  Playwright (Chromium). Screenshots: `docs/screenshots/`.
- The real-TF.js check needs Node.js; `tools/tfjs_check/` holds `@tensorflow/tfjs` for it.

## 8. Troubleshooting

| Problem | What to do |
|---|---|
| "No dataset chosen" | On that page: **Use the HAND TRACKER folder**, **Browse...** to a folder, or upload a .zip. |
| The step list on the left is gone | Click **Show steps** at the top left. |
| Live test: "Could not open camera" | Close Zoom/Teams/the Camera app (and the recorder). Try another camera number or the other backend in Settings. |
| The live window opens behind the browser | Click it in the taskbar ("HUDYAT Trainer - Live test"). |
| A job says it "stopped unexpectedly" | The black window was closed or the laptop slept. Start it again; finished parts are kept. |
| The app doesn't open | Run `SETUP (run once).bat` again, then `START HUDYAT TRAINER.bat`. |
| Many clips with low hand detection | Watch them on the Preprocess page; usually hands left the frame or the light was poor. Re-record them in the recorder. |
