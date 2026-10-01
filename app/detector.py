"""Object detection with OpenCV's DNN module (YOLOv4-tiny, COCO classes)."""
from dataclasses import dataclass, asdict

import cv2

from config import Config


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple          # (x, y, w, h) in pixels
    direction: str      # "left" | "ahead" | "right"
    distance: str       # "very close" | "close" | "far"

    def to_dict(self):
        return asdict(self)


def direction_from_box(box, frame_width):
    """Split the frame into thirds and report where the object's centre falls."""
    x, _, w, _ = box
    centre = (x + w / 2) / frame_width
    if centre < 1 / 3:
        return "left"
    if centre > 2 / 3:
        return "right"
    return "ahead"


def distance_from_box(box, frame_shape):
    """Rough proximity guess from how much of the frame the box covers.

    This is a heuristic (no depth sensor): a bigger box usually means a closer object.
    """
    _, _, w, h = box
    frame_h, frame_w = frame_shape[:2]
    ratio = (w * h) / float(frame_w * frame_h)
    if ratio > 0.25:
        return "very close"
    if ratio > 0.08:
        return "close"
    return "far"


class ObjectDetector:
    def __init__(self, model_dir=Config.MODEL_DIR):
        cfg = model_dir / "yolov4-tiny.cfg"
        weights = model_dir / "yolov4-tiny.weights"
        names = model_dir / "coco.names"
        missing = [p.name for p in (cfg, weights, names) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing)}. Run `python download_models.py`."
            )
        self.classes = names.read_text().strip().splitlines()
        net = cv2.dnn_DetectionModel(str(cfg), str(weights))
        net.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)
        self.model = net

    def detect(self, frame):
        class_ids, scores, boxes = self.model.detect(
            frame, Config.CONF_THRESHOLD, Config.NMS_THRESHOLD
        )
        results = []
        if len(class_ids) == 0:
            return results
        for cid, score, box in zip(class_ids.flatten(), scores.flatten(), boxes):
            box = tuple(int(v) for v in box)
            results.append(
                Detection(
                    label=self.classes[int(cid)],
                    confidence=round(float(score), 2),
                    box=box,
                    direction=direction_from_box(box, frame.shape[1]),
                    distance=distance_from_box(box, frame.shape),
                )
            )
        return results


def draw_detections(frame, detections):
    for d in detections:
        x, y, w, h = d.box
        colour = (60, 60, 230) if d.distance == "very close" else (60, 200, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        text = f"{d.label} {int(d.confidence * 100)}%"
        cv2.putText(frame, text, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    return frame
