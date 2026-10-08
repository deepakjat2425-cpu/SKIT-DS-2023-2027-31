"""Text-to-speech using gTTS. Files are cached by text+language hash."""
import hashlib

from config import Config


def speak(text, lang="en"):
    """Return the filename (inside static/audio) of an MP3 for `text`.

    Raises RuntimeError if gTTS or the network is unavailable; the frontend
    then falls back to the browser's built-in speech synthesis.
    """
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError("gTTS is not installed") from exc

    Config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(f"{lang}:{text}".encode("utf-8")).hexdigest()[:16] + ".mp3"
    path = Config.AUDIO_DIR / name
    if not path.exists():
        try:
            gTTS(text=text, lang=lang).save(str(path))
        except Exception as exc:  # network errors, unsupported language, ...
            raise RuntimeError(f"Speech generation failed: {exc}") from exc
    return name
