"""
Builds the extended-vocabulary detector used by AMCS (yoloe-26s-amcs.pt).

YOLOE is an open-vocabulary detector: it is told which objects to look for as text. Encoding that
text needs the MobileCLIP text model (~240 MB) and the `clip` package, but only once - the
resulting class embeddings are saved inside the .pt file, so the app itself never loads them.

Usage (from the project folder):
    python tools/build_extended_model.py            # writes yoloe-26s-amcs.pt next to app.py
    python tools/build_extended_model.py --work D:/tmp   # where to download the large text model

Re-run after editing modules/extended_vocabulary.py.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from modules.extended_vocabulary import EXTENDED_CLASS_NAMES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="yoloe-26s-seg.pt", help="YOLOE base weights (downloaded if missing)")
    parser.add_argument("--out", default=str(PROJECT_DIR / "yoloe-26s-amcs.pt"))
    parser.add_argument("--work", default=str(PROJECT_DIR / ".yoloe_build"),
                        help="folder for downloaded base weights and the text encoder")
    args = parser.parse_args()

    from ultralytics import YOLO

    out = Path(args.out).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)                     # Ultralytics downloads into the current folder

    model = YOLO(args.base)
    print(f"Encoding {len(EXTENDED_CLASS_NAMES)} class names...")
    model.set_classes(EXTENDED_CLASS_NAMES, model.get_text_pe(EXTENDED_CLASS_NAMES))
    model.save(str(out))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
