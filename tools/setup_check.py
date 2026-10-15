"""Run by 'SETUP (run once).bat': copy/check the MediaPipe model and check that everything imports.

Exit code 0 = ready. Messages are for the team, in plain language.
"""

import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import settings as cfg  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ok = True
    model = cfg.MODEL_PATH
    if not model.exists() or sha256(model) != cfg.MODEL_SHA256:
        sources = [d for d in (cfg.data_dir(), cfg.DEFAULT_DATA_DIR) if d]
        for d in sources:
            src = Path(d) / "models" / "hand_landmarker.task"
            if src.exists() and sha256(src) == cfg.MODEL_SHA256:
                model.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, model)                        # read from the recorder, written here
                print(f"Copied the MediaPipe hand model from {src}")
                break
    if model.exists() and sha256(model) == cfg.MODEL_SHA256:
        print("MediaPipe hand model: OK (checksum matches the recorder's README)")
    else:
        ok = False
        print("PROBLEM: models\\hand_landmarker.task is missing or different from the recorder's.")
        print("  Copy it from 'HUDYAT - HAND TRACKER\\models\\' into this folder's 'models' folder,")
        print("  then run SETUP again.")
    try:
        import cv2  # noqa: F401
        import keras  # noqa: F401
        import mediapipe  # noqa: F401
        import numpy  # noqa: F401
        import pyarrow  # noqa: F401
        import streamlit  # noqa: F401
        import tensorflow as tf
        print(f"Python packages: OK (TensorFlow {tf.__version__}, MediaPipe {mediapipe.__version__}, "
              f"Streamlit {streamlit.__version__})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"PROBLEM: a Python package does not load: {type(e).__name__}: {e}")
    if cfg.DEFAULT_DATA_DIR.exists() and not cfg.data_dir():
        print(f"Found the data-collection folder at {cfg.DEFAULT_DATA_DIR} (confirm it in the app's Settings).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
