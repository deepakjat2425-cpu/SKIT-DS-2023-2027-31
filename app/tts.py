"""Text-to-speech using gTTS. Audio files are cached by text and language."""
import hashlib
from config import Config

# gTTS language codes supported by this project. Keep in sync with the UI.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
}

def speak(text, lang="en"):
    """Return an MP3 filename. Raises RuntimeError if speech generation fails."""
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}")
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
        except Exception as exc:
            raise RuntimeError(f"Speech generation failed: {exc}") from exc
    return name
