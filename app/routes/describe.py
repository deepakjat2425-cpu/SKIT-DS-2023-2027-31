from flask import Blueprint, jsonify, request, url_for
from app.camera import stream
from app.describer import describe
from app.tts import speak, SUPPORTED_LANGUAGES

bp = Blueprint("describe", __name__)

@bp.route("/api/describe", methods=["POST"])
def describe_scene():
    data = request.get_json(silent=True) or {}
    lang = data.get("lang", "en")
    if not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES:
        return jsonify(
            error="Unsupported language",
            supported_languages=SUPPORTED_LANGUAGES,
        ), 400

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
        language_name=SUPPORTED_LANGUAGES[lang],
        audio_url=audio_url,
        tts_error=tts_error,
    )
