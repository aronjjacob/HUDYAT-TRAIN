# HUDYAT Trainer

Turns the FSL clips recorded with the data-collection tool (**HUDYAT - HAND TRACKER**) into sign
models: landmarks → two honest tests → a trained model → TensorFlow.js for the browser app → a live
webcam test → figures and tables for the paper. One model per category.

The data-collection folder is **only read**. Everything the trainer makes goes into this folder's
`work\` folder.

---

## 1. Setup (once per laptop)

1. Install **Python 3.12 (64-bit)** from python.org if it isn't installed yet (the same one the
   recorder uses).
2. Put this folder somewhere short, e.g. `C:\HUDYAT - TRAINER` or `Documents\HUDYAT - TRAINER`.
   (Very deep folders break the install: Windows' long-path limit.)
3. Double-click **`SETUP (run once).bat`**. It creates the Python environment (`venv\`), installs
   the packages from `requirements.txt`, and checks the MediaPipe hand model. The first run takes
   5–15 minutes.
4. Double-click **`START HUDYAT TRAINER.bat`**. A black window opens (keep it open while you work)
   and the app opens in the browser at http://localhost:8517. Close the black window to stop the app.

The app finds `Documents\HUDYAT - HAND TRACKER` by itself and uses its recordings.

**Desktop shortcut:** right-click `START HUDYAT TRAINER.bat` → *Show more options* → *Send to* →
*Desktop (create shortcut)*. If the app is already running, the shortcut just opens the browser again.

---

## 2. The pages

The sidebar holds the **Dataset** in use (switch datasets here), the six pages numbered in the usual
order (Data → Preprocess → Test the model → Train model → Live test → Results for the paper) and
**⚙ Settings**. A step done at least once has a ✓, and **Next:** names the first step not done yet;
a finished page ends with a link to the next one. Every page can still be opened at any time; a page
that needs something first (e.g. landmarks) says so in plain words. Buttons that can't run yet are
greyed out with the reason written under them. Each page has its own **Category** picker; the choice
is remembered when you move between pages. Pages show the main result first; details sit in tabs
(Test the model, Results for the paper) or folded boxes.

| | Page | What you do there |
|---|---|---|
| 1 | **Data** | See what was recorded: signers (with their signing hand), signs, kept clips, clips per sign × signer (S1–S7, "—" = not recorded yet) and plain warnings (missing signers, re-record marks, low hand detection). **Look at a clip**: the video next to its landmark skeleton; **Reject** it with a reason (it is then left out of every test and model) or **Restore** it. **Add a dataset**: a .zip / folder (see 2.1). |
| 2 | **Preprocess** | Choose the **categories to extract** (all, or e.g. only `greetings`). **Extract landmarks**: MediaPipe on every new kept clip of those categories (in parallel). Shows clips per category (extracted / to extract), hands per signer (left/right detected vs. the recorded signing hand), the trim summary and low-detection clips. *More*: re-extract the chosen categories. *Advanced*: detection threshold, trim margin, MediaPipe confidences. |
| 3 | **Test the model** | **Test 1** (random split) and **Test 2** (leave-one-signer-out), see section 4. *Settings for this run*: canonical hand, mirror augmentation, epochs, augmentation, batch, seed. Results: PASS/FAIL with the failed checks, the main numbers and per round, then tabs for recall per sign, confusion matrix, training curves, the left-hand check, **Compare runs** and details. A stopped run can be **Resumed** (finished rounds are kept). |
| 4 | **Train model** | Pick a category, or **All categories** (one model per category, trained one after another), and press **Train**: every signer's extracted clips, 15 % kept back only to know when to stop. No test run is needed. *Training settings*: canonical hand, mirror augmentation, epochs, seed, batch, augmentation; or **use the settings of a test run**. Then **Export for the app** (the HUDYAT app's model files, checked against the trained model), or tick several models and export them together. Training also remakes that category's paper results. |
| 5 | **Live test** | Opens a camera window in the recorder's style. **Free** mode shows the top 3 guesses; **Test** mode scores a tester on target signs and saves every attempt. Set the tester's signing hand; *Advanced* has the Hand setting (Auto / same as the tester) and, for app models, the capture. |
| 6 | **Results for the paper** | **Generate figures**: 300-dpi PNG + PDF figures, tables, captions and a results summary with run IDs, plus a ZIP of everything, shown in tabs (Tables · Figures · Summary). Figures whose data is missing are listed as "Not made yet". |

Only one long job (extract, test, training, export, live test, figures) runs at a time. A job
keeps running if you refresh the page or close the tab; reopen the page to see its progress.
**Stop** ends it (finished parts are kept). Errors appear in plain language with what to do; the
*Full log* box under the message has the details for the developers.

### 2.1 Other datasets

**Dataset** in the sidebar switches between your HAND TRACKER recordings and uploaded
datasets; **Data → Add a dataset** adds one. Each dataset keeps its own samples, tests and models (uploads: `work\imports\<name>\work`).
To add one, upload a **.zip** or pick a **folder**. The app first shows what it found (layout,
signers, signs, clips, problems); nothing is copied until you press **Import**. Accepted:

- **a copy of a HUDYAT - HAND TRACKER folder** (signs.csv + dataset). A real HAND TRACKER folder
  is used where it is, read only.
- **videos per signer**: `category\sign\S1…S7\file.mp4`, like the recorder's `dataset\raw`.
- **photos, one folder per sign** (e.g. the Kaggle FSL alphabet `A\*.jpg`, `B\*.jpg` …). A photo
  becomes a held-still 32-frame sample. Photos have no signer IDs, so **only Test 1** is possible.

*More: combine datasets* on the Data page joins datasets into one model (e.g. A–Y photos + J and Z
videos). The combined model has only the chosen signs (no `none` class).

---

## 3. Which clips are used

- Only our signers **S1–S7**; the imported public `PUB` clips are ignored.
- Only `status = kept`, not `expert_status = rejected` (if that column exists), not rejected on the
  Data page.
- Hand detection inside the sign window **≥ 80 %** (Settings).
- `alphabet_static_test` is not trained here (it is the test set of the static A–Y photo model).
- **There is no `none` class.** A model has exactly one output per sign of its category (N signs → N
  outputs). If a dataset has a `none` category, it is ignored: not extracted, not trained, not shown.
  "Not a sign" is handled in the live test instead: the 85 % confidence threshold and the sign window
  (see 6.4).

---

## 4. Test 1 and Test 2

| | Test 1: random split | Test 2: leave-one-signer-out (LOSO) |
|---|---|---|
| Question | Did the model learn the signs at all? | Will it work for someone it has never seen? |
| How | Per sign, seeded 70 / 15 / 15 % of all clips of all signers. | One round per signer: test = that signer only; train / validation (15 %, for early stopping only) = everyone else. 3 rounds now, 7 when S1–S7 are recorded. |
| Limitation | Test clips come from the same people the model trained on, so it does not show how HUDYAT works for new people. | With few signers the spread between rounds is large. |

**Test 2 is the headline number.** PASS = mean accuracy and macro precision / recall / F1 ≥ 85 %
across the rounds **and** every sign's recall ≥ 70 %. Results are mean ± SD (sample SD) over the rounds.

**No leakage:** augmentation and mirrored copies are made only from the training clips; validation
and test clips are never augmented.

**An honest caveat for the paper.** If settings are chosen by looking at Test 2 (e.g. mirror augmentation on or off),
Test 2 has been used for tuning, and with 3 signers nobody is left over for an untouched check.
Say so in the paper. When S4–S7 are recorded, their rounds are the cleaner number.

Runs are saved in `work\experiments\<test>\<category>\<run_id>\` (`config.json` with every setting
and the clips per set per round, `results.json`, one folder per round).

---

## 5. Left-handed users

All our signers are right-handed. A left-handed learner signs the mirror image, which the model
has never seen.

- **Canonical hand (on by default).** Before the model sees a sample, a left-handed sample is turned
  into its right-handed mirror image (x → −x, Left and Right swapped). The model only learns
  right-dominant signing. Training clips use the recorder's `signing_hand`. In use, the **Hand**
  setting is Right / Left / **Auto**; Auto decides per sign: one hand → that hand is dominant; two
  hands → the one whose wrist moves more; otherwise right.
- **Mirror augmentation (off by default).** Mirrors a share of the training clips. FSL experts must
  confirm that mirrored signs are still correct signs first; signs marked `mirror_ok = 0` are
  excluded once any sign has `mirror_ok = 1`.
- **Left-hand check in every test.** The test clips are scored as recorded and mirrored, each
  without and with the canonical mirror (Auto), and the page says how often Auto picked the right hand.
  Mirrored clips only *simulate* left-handed users; **real left-handed testers are still needed.**
- Every live-test attempt records the tester's signing hand.

Stored samples stay exactly as recorded (recorder README 6.1); the mirror is applied when samples are
loaded, so switching it needs no re-extraction.

---

## 6. How it works

### 6.1 Preprocessing (identical to the browser app)

Recorder README section 6: MediaPipe HandLandmarker (VIDEO mode, 2 hands, the same
`hand_landmarker.task`) on the **unmirrored** frames → per-hand normalization (aspect correction,
wrist at the origin, divided by the wrist–MCP9 distance) → sign window → 32 frames. The trim rule,
the handedness rule and the MediaPipe settings are imported from the recorder's own
`hudyat_common.py` (Settings shows this); if it cannot be imported, the trainer's copies are used,
and a test checks they give identical results.

Each sample stores 149 values per frame (the 128 of the recorder format + each hand's raw wrist
and size + where each hand is against the face, recorder README 6.7). Extraction runs the MediaPipe face
model too (`models\face_landmarker.task`, or the recorder's copy) and writes `face_detect_rate` per clip;
clips where the face was found in < 80 % of the sign are listed in the log as retake candidates. The model
input is **128 values per frame** (Left 63 + present flag, Right 63 + flag),
normalized exactly like the HUDYAT app does it (`hand-relative-v1`: per hand, relative to the wrist,
divided by the 2D wrist → middle-finger MCP length), so a model trained here sees the same numbers
in the app. Three feature versions (*Model input* in the training settings, `--feature-version`): **v1**
(`hand-relative-v1`, the wrist is always 0), **v2** (`hand-relative-sequence-v1`, the format of Russel's
trainer: v1 plus the wrist's movement since the sign started) and **v3** (`hand-face-location-v1`, 141
values: v1 plus where the hand is against the face, so Father at the forehead and Mother at the chin differ).
v3 is for Family; its export also names the face model, which the app loads only for v3 models. All
come from the stored samples, so switching needs no new extraction; compare them with the same seed. The exact definitions and a JavaScript reference are
in [`docs/FEATURES.md`](docs/FEATURES.md).

### 6.2 Training

Augmentation on the training set only, in memory, fixed seed: noise, scale ±10 %, rotation ±15°
about the wrist, time warp, frame dropout; no shift; an absent
hand stays zero. Model: Conv1D 64 → BN → MaxPool → Dropout → Conv1D 128 → BN → Dropout → BiLSTM 128
→ Dropout → BiLSTM 64 → Dropout → Dense 64 → softmax (the app's network, layer for layer; BN epsilon 0.0001). Adam 1e-3, rare signs weighted more, early
stopping on validation loss (best weights restored), learning-rate reduction on plateau.

### 6.3 Export

`work\exports\<category>-dabest-<dataset>-v1\`, the files the HUDYAT app loads
(`server/model-assets/models/<category>/<version>/`): **`model.json`** (format `fsl-conv1d-bilstm-v1`:
classes, preprocessing, where each weight is), **`weights.bin`**, **`parity.json`** (real test clips and
the trained model's answers, for the app's model test), plus `preprocessing.json`, `catalog_entry.json`
(the entry for the app's `catalog.json`), `verify.json` (this export's check) and a `README.txt` that
says where each file goes. The app stores the hands at other indices than the trainer; the export
reorders the model's first layer for that, so the app gets exactly the trained model's answers.

### 6.3b Share the dataset (landmarks only)

Preprocess → **Export** on a category's card copies its training samples into
`work\datasets\<category>-<dataset>-landmarks-<date>\` and a `.zip` of it. It holds:
- the `.npy` samples
- `dataset\metadata.csv`: signer, signing hand, sign and take, per sample
- `landmarks_index.csv`
- `signs.csv`, `signers.csv`
- `excluded.csv`: what was left out and why
- `dataset.json`
- `FEATURES.md`: what the 149 values are
- `README.txt`

For training only: no videos, no model, no review log. Only samples in the current format are included, so
Extract each category first.

**Open a shared dataset:** upload the `.zip` (or add the folder) on the Data page. It opens as *landmarks only*
and goes straight to Test the model and Train model. Extracting is switched off, because there are no videos.
It is one more entry under **Dataset** in the sidebar, with its own `work\`. The recordings dataset keeps
working as before.

### 6.4 Live test

The camera window finds each sign itself (README 6.6): it starts when a hand appears (with the 0.1 s
before it) and ends when no hand has been seen for 0.2 s or after 3 s. A prediction is only made for
such a **completed** sign window, never while the hands are idle: nothing while no hand is in view,
nothing for a detection blip (under 0.2 s of hands), and after a 3-second window the hands must go down
before the next sign counts ("Lower your hands, then sign again"). The preview is mirrored like a
selfie; MediaPipe runs on the unmirrored frames. Below **85 %** confidence (Settings → *Live test:
minimum confidence*) it shows "Not sure, try again". In Test mode an attempt is correct only when the
guess is the target **and** its confidence is at least that threshold. Response time
= end of the sign window → prediction shown (target: under 1 second); every attempt also records the time from
the hand appearing to the prediction.

**Recognizer: HUDYAT app model.** After *Export for the app*, the live test can run the exported files
themselves (the app's network, input layout and rule: at least 80 % sure and 15 points ahead of the second
guess), with two captures: **Fixed** (the sign ends when the hands go down, no blending with a missing hand,
left hands mirrored) and **As the app today** (a fixed 2-second clip). Run both with the same tester; the
Live results show accuracy and both response times per setup. [`docs/APP_RECOGNITION.md`](docs/APP_RECOGNITION.md)
is the write-up for the app team, with the Fixed capture in JavaScript (tested against the trainer). Keys: **Space** pause / start,
**N** next sign, **H** hand, **M** mode, **Q** quit (the buttons on screen do the same).

---

## 7. Where the outputs go

```
work\
├── landmarks\<category>\<sign_id>\<signer>\<file>.npy   samples (32 frames × 149 values)
├── landmarks_index.csv                                  one row per sample (trim, detection rates, status)
├── validation.csv                                       Reject / Restore decisions (last one counts)
├── experiments\<random|loso>\<category>\<run_id>\       Test 1 / Test 2 runs
├── models\<category>\                                   trained model: model.keras, labels.json, config.json,
│                                                        preprocess.json, history.csv
├── exports\<category>-dabest-<dataset>-v1\              the export for the HUDYAT app
├── datasets\<category>-<dataset>-landmarks-<date>\      a category's samples to share (+ .zip), section 6.3b
├── live_tests\<category>.csv                            every Test-mode attempt (with tester_hand)
├── paper\<category>\                                    figures\, tables\ (Table 6 / 7 and details), captions.md,
│                                                        results_summary.md, paper.zip
├── imports\<name>\                                      uploaded datasets, each with its own work\
├── preview_cache\                                       browser-playable copies of clips (safe to delete)
└── jobs\                                                logs and status of the background jobs
```

`trainer_settings.json` (this folder) holds the dataset in use, camera, thresholds and default hand.

**Upgrading from v1:** samples made by v1 have 128 values per frame. The first **Extract landmarks**
re-extracts them all once (about 1,260 clips).

---

## 8. Without the app (command line)

Every button runs one of these jobs (`venv\Scripts\activate` first; `--help` lists each job's options):

```
python -m trainer.runner extract [--all] [--category greetings ...]
python -m trainer.runner experiment --test random|loso --category colors|all [--no-canonical]
                     [--mirror-aug] [--epochs 150 --batch 32 --aug 5 --seed 42] [--resume RUN_ID]
python -m trainer.runner train --category colors [greetings ...]|all [--from-run RUN_ID]   one model per category
                     [--no-canonical] [--epochs 150 --batch 32 --aug 5 --seed 42]
python -m trainer.runner export --category colors [greetings ...]|all       files for the HUDYAT app
python -m trainer.runner dataset --category colors [greetings ...]|all      the landmark dataset (.npy) to share
python -m trainer.runner live --category colors [--mode test --tester NAME --tester-hand left --hand auto --target red]
                     [--recognizer app --capture fixed|app]
python -m trainer.runner paper --category colors [--test1 RUN_ID] [--test2 RUN_ID]
```

---

## 9. Tests (for the developers)

```
venv\Scripts\python -m pytest -m "not slow"     unit tests (about 5 minutes)
venv\Scripts\python -m pytest                   + end-to-end and UI tests (about 30 minutes)
```

- **Unit tests:** landmark format pinned to README section 6, parity with `hudyat_common.py`,
  the model input = the app's hand-relative-v1 (including the JavaScript reference run in Node), mirror / canonical / Auto, Test 1 and
  LOSO splits (no clip in two sets, held-out signer never trained on, resume), no leakage, metrics and
  PASS/FAIL, figures, the paper tables, `preprocess.json`, the app export (loaded with the app's own
  `buildModel` / `loadWeights` in real TF.js), upload scans, the job lock and the live sign window.
- **End-to-end** (`tests/test_end_to_end.py`): a temporary dataset of real S1–S3 clips, with S3
  marked left-handed and its videos mirrored, run through Preprocess → Test 1 → Test 2 →
  Train model → Export for the app (its loader vs Keras) → live `--video` with Hand = Auto → Paper. Its numbers are only a
  pipeline check.
- **UI** (`tests/test_ui.py`, `tests/test_ui_upload.py`): starts the app headless and clicks through
  every page with Playwright, including uploading a photo .zip (Scan → Import → Test 1 → Train
  model → Export). Screenshots: `docs/screenshots/v2/`.

---

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| Setup: "Installing the packages failed" with *long path* | Move the folder closer to `C:\`, delete `venv\` inside it, run SETUP again. |
| "No dataset chosen" | On that page: **Use the HAND TRACKER recordings**, add a folder, or upload a .zip. |
| Test 2 is greyed out | It needs clips of at least two signers (photo datasets: Test 1 only). |
| Live test: "Could not open camera" | Close Zoom/Teams/the Camera app (and the recorder). Try another camera number or backend in Settings. |
| The live window opens behind the browser | Click it in the taskbar ("HUDYAT Trainer - Live test"). |
| A job says it "stopped unexpectedly" | The black window was closed or the laptop slept. Start it again (tests: **Resume**); finished parts are kept. |
| The app doesn't open | Run `SETUP (run once).bat` again, then `START HUDYAT TRAINER.bat`. |
| Many clips with low hand detection | Watch them on the Data page; usually the hands left the frame or the light was poor. Re-record them. |
