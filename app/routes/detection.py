from flask import Blueprint, Response, jsonify

from app.camera import stream

bp = Blueprint("detection", __name__)


@bp.route("/video_feed")
def video_feed():
    return Response(stream.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/api/detections")
def detections():
    stream.start()
    return jsonify(
        detections=[d.to_dict() for d in stream.detections],
        error=stream.error,
    )
