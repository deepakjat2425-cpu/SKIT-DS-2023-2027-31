"""
Speech Synthesis Module using gTTS (Google Text-to-Speech) with pyttsx3 & Web Speech fallback.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import os
import hashlib
import threading
import time
from pathlib import Path
from gtts import gTTS
import config

class SpeechSynthesizer:
    def __init__(self, cache_dir=config.AUDIO_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_spoken_time = 0
        self.last_spoken_hash = None
        self.cooldown = config.SPEECH_COOLDOWN_SECONDS
        self._offline_engine = None
        # When gTTS fails (no internet), skip it for a while instead of
        # stalling every request on a network timeout.
        self._gtts_disabled_until = 0.0
        # Audio being made in the background for the live loop: file name -> threading.Event
        self._jobs = {}
        self._jobs_lock = threading.Lock()
        self._prune_cache()

    def synthesize_later(self, text, lang="en"):
        """
        Like synthesize(), but returns at once. gTTS needs about 2 s for a new sentence; the live
        loop must not wait for it, so the audio is made in the background and the returned URL
        (/api/speech/<file>) answers as soon as it is ready. Returns None when server speech is
        unavailable for now (the browser then speaks with its own voice).
        """
        if not text or not text.strip():
            return None
        clean_text = text.strip()
        filename = self._get_cache_filename(clean_text, lang)
        file_path = self.cache_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            return f"/static/audio_cache/{filename}"
        if time.time() < self._gtts_disabled_until:
            return None
        with self._jobs_lock:
            if filename not in self._jobs:
                done = threading.Event()
                self._jobs[filename] = done
                threading.Thread(target=self._make_in_background, args=(clean_text, lang, filename, done),
                                 daemon=True, name="tts").start()
        return f"/api/speech/{filename}"

    def _make_in_background(self, text, lang, filename, done):
        try:
            self.synthesize(text, lang=lang, force=True)
        finally:
            done.set()
            with self._jobs_lock:
                self._jobs.pop(filename, None)

    def wait_for(self, filename, timeout=10.0):
        """Path of a finished audio file, waiting for its background job if one is running; else None."""
        with self._jobs_lock:
            done = self._jobs.get(filename)
        if done is not None:
            done.wait(timeout)
        file_path = self.cache_dir / filename
        return file_path if file_path.exists() and file_path.stat().st_size > 0 else None

    def _prune_cache(self):
        """Keeps the MP3 cache bounded by deleting the oldest files."""
        try:
            files = sorted(self.cache_dir.glob("speech_*.mp3"), key=lambda p: p.stat().st_mtime)
            excess = len(files) - config.AUDIO_CACHE_MAX_FILES
            for old in files[:max(0, excess)]:
                old.unlink(missing_ok=True)
        except OSError as e:
            print(f"[SpeechSynthesizer] Cache prune skipped: {e}")

    def _get_cache_filename(self, text, lang):
        """Generates deterministic filename using MD5 hash of text and language."""
        raw_key = f"{lang}_{text.strip()}".encode('utf-8')
        md5_hash = hashlib.md5(raw_key).hexdigest()
        return f"speech_{md5_hash}.mp3"

    def synthesize(self, text, lang="en", force=False):
        """
        Synthesizes text into spoken audio in the selected language.
        Returns:
            dict containing:
                - audio_url: URL to the generated MP3 file (relative to /static)
                - text: The spoken text
                - lang: The language code
                - played_status: 'generated', 'cached', or 'skipped_cooldown'
                - fallback_browser: boolean whether client should speak via Web Speech API
        """
        if not text or not text.strip():
            return {"audio_url": None, "text": "", "lang": lang, "status": "empty"}

        clean_text = text.strip()
        now = time.time()
        text_hash = hashlib.md5(f"{lang}_{clean_text}".encode('utf-8')).hexdigest()

        # Cooldown check: do not repeat exact same text unless cooldown has elapsed or forced
        if not force and text_hash == self.last_spoken_hash and (now - self.last_spoken_time) < self.cooldown:
            return {
                "audio_url": None,
                "text": clean_text,
                "lang": lang,
                "status": "skipped_cooldown",
                "fallback_browser": False
            }

        filename = self._get_cache_filename(clean_text, lang)
        file_path = self.cache_dir / filename
        relative_url = f"/static/audio_cache/{filename}"

        # 1. Return cached audio file if already generated
        if file_path.exists() and file_path.stat().st_size > 0:
            self.last_spoken_time = now
            self.last_spoken_hash = text_hash
            return {
                "audio_url": relative_url,
                "text": clean_text,
                "lang": lang,
                "status": "cached",
                "fallback_browser": False
            }

        # 2. Generate with gTTS (unless it failed recently)
        if now < self._gtts_disabled_until:
            self.last_spoken_time = now
            self.last_spoken_hash = text_hash
            return {
                "audio_url": None,
                "text": clean_text,
                "lang": lang,
                "status": "browser_fallback",
                "fallback_browser": True
            }
        try:
            # Map language code if necessary
            gtts_lang = lang.split("-")[0] if "-" in lang and lang not in ["zh-CN", "zh-TW"] else lang
            tts = gTTS(text=clean_text, lang=gtts_lang, slow=False, timeout=config.GTTS_TIMEOUT_SECONDS)
            # Write to a temp file first so a failed download never leaves a broken MP3 in the cache
            tmp_path = file_path.with_suffix(".part")
            tts.save(str(tmp_path))
            os.replace(tmp_path, file_path)
            if len(list(self.cache_dir.glob("speech_*.mp3"))) > config.AUDIO_CACHE_MAX_FILES:
                self._prune_cache()

            self.last_spoken_time = now
            self.last_spoken_hash = text_hash
            return {
                "audio_url": relative_url,
                "text": clean_text,
                "lang": lang,
                "status": "generated",
                "fallback_browser": False
            }
        except Exception as e:
            print(f"[SpeechSynthesizer] gTTS synthesis error: {e}. Falling back to browser speech.")
            self._gtts_disabled_until = now + config.GTTS_RETRY_AFTER_SECONDS
            try:
                file_path.with_suffix(".part").unlink(missing_ok=True)
            except OSError:
                pass
            # If gTTS fails (e.g. offline/network blocked), instruct browser client to use Web Speech API
            self.last_spoken_time = now
            self.last_spoken_hash = text_hash
            return {
                "audio_url": None,
                "text": clean_text,
                "lang": lang,
                "status": "browser_fallback",
                "fallback_browser": True
            }
