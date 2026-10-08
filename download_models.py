"""Download the YOLOv4-tiny (COCO) files used by the detector into ./models.

Run once:  python download_models.py
"""
import urllib.request

from config import Config

FILES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
}


def main():
    Config.MODEL_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = Config.MODEL_DIR / name
        if target.exists():
            print(f"✓ {name} already present")
            continue
        print(f"Downloading {name} …")
        urllib.request.urlretrieve(url, target)
    print("Done.")


if __name__ == "__main__":
    main()
