from flask import Blueprint, jsonify, request, url_for

from app.camera import stream
from app.describer import describe
from app.tts import speak

bp = Blueprint("describe", __name__)


@bp.route("/api/describe", methods=["POST"])
def describe_scene():
    data = request.get_json(silent=True) or {}
    lang = data.get("lang", "en")

    result = describe([d.to_dict() for d in stream.detections])

    audio_url, tts_error = None, None
    try:
        audio_url = url_for("static", filename=f"audio/{speak(result['text'], lang)}")
    except RuntimeError as exc:
        tts_error = str(exc)

    return jsonify(
        text=result["text"],
        alerts=result["alerts"],
        lang=lang,
        audio_url=audio_url,
        tts_error=tts_error,
    )
