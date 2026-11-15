"""
OCR Text Detection, Extraction and Multilingual Translation Engine.
Enables sign reading, board translation, and multilingual scene narration.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np
import os
import re
import time
from deep_translator import GoogleTranslator, MyMemoryTranslator
import config

# Fallback offline dictionary for common navigational and assistive terms
OFFLINE_DICTIONARY = {
    "The path ahead is clear. No obstacles detected.": {
        "hi": "आगे का रास्ता साफ है। कोई बाधा नहीं दिखी।",
        "es": "El camino por delante está despejado. No se detectaron obstáculos.",
        "fr": "La voie devant vous est libre. Aucun obstacle détecté.",
        "de": "Der Weg nach vorne ist frei. Keine Hindernisse erkannt.",
        "zh-CN": "前方道路畅通。未检测到障碍物。",
        "ja": "前方の通路はクリアです。障害物は検出されませんでした。",
        "ar": "الطريق أمامك واضح. لم يتم اكتشاف أي عوائق.",
        "ru": "Путь впереди свободен. Препятствий не обнаружено.",
        "bn": "সামনের পথ পরিষ্কার। কোনো বাধা পাওয়া যায়নি।",
        "ta": "முன்னால் உள்ள பாதை தெளிவாக உள்ளது. தடைகள் ஏதும் இல்லை.",
        "te": "ముందు ఉన్న మార్గం స్పష్టంగా ఉంది. ఎటువంటి అడ్డంకులు లేవు.",
        "mr": "पुढील रस्ता मोकळा आहे. कोणताही अडथळा आढळला नाही.",
        "gu": "આગળનો રસ્તો સાફ છે. કોઈ અવરોધ જણાયો નથી."
    },
    "Caution!": {
        "hi": "सावधान!",
        "es": "¡Precaución!",
        "fr": "Attention!",
        "de": "Achtung!",
        "zh-CN": "小心！",
        "ja": "注意！",
        "ar": "احترس!",
        "ru": "Осторожно!",
        "bn": "সাবধান!",
        "ta": "எச்சரிக்கை!",
        "te": "జాగ్రత్త!",
        "mr": "सावध रहा!",
        "gu": "સાવચેત રહો!"
    },
    "Railway Station": {
        "hi": "रेलवे स्टेशन",
        "es": "Estación de tren",
        "fr": "Gare ferroviaire",
        "de": "Bahnhof",
        "zh-CN": "火车站",
        "ja": "鉄道駅",
        "ar": "محطة القطار",
        "ru": "Железнодорожный вокзал",
        "bn": "রেলওয়ে স্টেশন",
        "ta": "ரயில் நிலையம்",
        "te": "రైల్వే స్టేషన్",
        "mr": "रेल्वे स्थानक",
        "gu": "રેલ્વે સ્ટેશન"
    },
    "Hospital": {
        "hi": "अस्पताल",
        "es": "Hospital",
        "fr": "Hôpital",
        "de": "Krankenhaus",
        "zh-CN": "医院",
        "ja": "病院",
        "ar": "مستشفى",
        "ru": "Больница",
        "bn": "হাসপাতাল",
        "ta": "மருத்துவமனை",
        "te": "ఆసుపత్రి",
        "mr": "रुग्णालय",
        "gu": "હોસ્પિટલ"
    },
    "Exit": {
        "hi": "निकास",
        "es": "Salida",
        "fr": "Sortie",
        "de": "Ausgang",
        "zh-CN": "出口",
        "ja": "出口",
        "ar": "مخرج",
        "ru": "Выход",
        "bn": "প্রস্থান",
        "ta": "வெளியேறு",
        "te": "నిష్క్రమణ",
        "mr": "बाहेर पडण्याचा मार्ग",
        "gu": "બહાર નીકળો"
    },
    "Entrance": {
        "hi": "प्रवेश द्वार",
        "es": "Entrada",
        "fr": "Entrée",
        "de": "Eingang",
        "zh-CN": "入口",
        "ja": "入口",
        "ar": "مدخل",
        "ru": "Вход",
        "bn": "প্রবেশপথ",
        "ta": "நுழைவாயில்",
        "te": "ప్రవేశం",
        "mr": "प्रवेशद्वार",
        "gu": "પ્રવેશ"
    },
    "Stop": {
        "hi": "रुकिए",
        "es": "Alto",
        "fr": "Arrêt",
        "de": "Halt",
        "zh-CN": "停止",
        "ja": "止まれ",
        "ar": "قف",
        "ru": "Стоп",
        "bn": "থামুন",
        "ta": "நில்",
        "te": "ఆగండి",
        "mr": "थांबा",
        "gu": "ઊભા રહો"
    }
}

# After both online translators fail (offline, or rate-limited), they are not tried again for this
# long; meanwhile the offline fallback answers at once instead of waiting about a second each time.
ONLINE_RETRY_SECONDS = 30


class OCRTranslator:
    _online_retry_at = 0.0

    def __init__(self):
        self.ocr_reader = None
        self._translation_cache = {}
        self._init_ocr()

    def _init_ocr(self):
        """Initializes EasyOCR reader with local storage directory."""
        try:
            import easyocr
            model_dir = config.BASE_DIR / ".easyocr_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"[OCRTranslator] Initializing EasyOCR (storage: {model_dir})...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(model_dir), user_network_directory=str(model_dir), verbose=False)
            print("[OCRTranslator] EasyOCR initialized successfully.")
        except Exception as e:
            print(f"[OCRTranslator] Warning: EasyOCR reader initialization deferred: {e}")
            self.ocr_reader = None

    def preprocess_for_ocr(self, image):
        """Preprocesses image to enhance text contrast and edge definitions."""
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Contrast adjustment via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def extract_text(self, frame):
        """
        Extracts visible text, signboards, and labels from an image frame.
        Uses EasyOCR if loaded; otherwise uses OpenCV morphological text region detector.
        Returns:
            list of dicts: [
                {
                    "text": str,
                    "confidence": float,
                    "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                }, ...
            ], combined_text: str
        """
        if frame is None:
            return [], ""

        # 1. Try EasyOCR if available
        if self.ocr_reader is not None:
            try:
                ocr_results = self.ocr_reader.readtext(frame)
                extracted_items = []
                clean_texts = []

                for (bbox, text, conf) in ocr_results:
                    clean_t = text.strip()
                    if len(clean_t) >= 2 and conf >= 0.25:
                        pts = [[int(pt[0]), int(pt[1])] for pt in bbox]
                        extracted_items.append({
                            "text": clean_t,
                            "confidence": round(float(conf), 2),
                            "bbox": pts
                        })
                        clean_texts.append(clean_t)

                if clean_texts:
                    return extracted_items, " | ".join(clean_texts)
            except Exception as e:
                print(f"[OCRTranslator] EasyOCR error: {e}")

        # 2. OpenCV Text Region & Signboard Contour Detector Fallback
        return self._detect_text_regions_cv(frame)

    def _detect_text_regions_cv(self, frame):
        """Detects prominent text / sign regions using OpenCV morphological processing."""
        h, w = frame.shape[:2]
        if len(frame.shape) == 2:
            gray = frame.copy()
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0].copy()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Morphological gradient to detect high contrast character edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Binarize
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Connect horizontal text lines
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, connect_kernel)
        
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter text-like aspect ratios
            aspect = bw / float(bh) if bh > 0 else 0
            if bw > 40 and bh > 15 and 1.2 <= aspect <= 12 and (bw * bh) < (h * w * 0.7):
                pts = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
                items.append({
                    "text": "Signboard Text",
                    "confidence": 0.85,
                    "bbox": pts
                })

        return items, "Detected Signboard / Text Region" if items else ""

    def draw_ocr_boxes(self, frame, ocr_items):
        """Draws bounding polygons and extracted text badges on frame."""
        annotated = frame.copy()
        for item in ocr_items:
            pts = np.array(item["bbox"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            top_pt = tuple(item["bbox"][0])
            label = f"TEXT: {item['text']}"
            cv2.putText(annotated, label, (max(0, top_pt[0]), max(15, top_pt[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def translate_text(self, text, target_language="en", source_language="auto"):
        """
        Translates text to target language with multi-tier fallback:
        1. Local translation cache
        2. GoogleTranslator via deep-translator
        3. MyMemoryTranslator fallback
        4. Offline navigational dictionary
        5. Original text if untranslatable
        """
        if not text or not text.strip():
            return ""

        clean_text = text.strip()
        target_lang = target_language.lower()

        # If source and target are same (e.g., en -> en)
        if target_lang == "en" and source_language == "en":
            return clean_text

        cache_key = (clean_text, target_lang)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        # 1. Check offline dictionary for exact or substring match
        if clean_text in OFFLINE_DICTIONARY:
            if target_lang in OFFLINE_DICTIONARY[clean_text]:
                result = OFFLINE_DICTIONARY[clean_text][target_lang]
                self._translation_cache[cache_key] = result
                return result

        # 2. Google Translator, then 3. MyMemory Translator (skipped for a while after both failed)
        if time.time() >= self._online_retry_at:
            for attempt in (self._translate_google, self._translate_mymemory):
                try:
                    translated = attempt(clean_text, source_language, target_lang)
                except Exception:
                    continue
                if translated and translated.strip():
                    self._translation_cache[cache_key] = translated.strip()
                    return translated.strip()
            self._online_retry_at = time.time() + ONLINE_RETRY_SECONDS

        # 4. Word-by-word offline fallback substitution
        translated_words = []
        for word in re.findall(r"[\w']+|[.,!?;]", clean_text):
            if word in OFFLINE_DICTIONARY and target_lang in OFFLINE_DICTIONARY[word]:
                translated_words.append(OFFLINE_DICTIONARY[word][target_lang])
            else:
                translated_words.append(word)

        # Not cached: once the online service is back, this sentence gets a proper translation
        return " ".join(translated_words)

    @staticmethod
    def _translate_google(text, source_language, target_lang):
        return GoogleTranslator(source=source_language, target=target_lang).translate(text)

    @staticmethod
    def _translate_mymemory(text, source_language, target_lang):
        mm_src = "en-GB" if source_language == "auto" else source_language
        mm_target = f"{target_lang}-IN" if target_lang in ["hi", "ta", "te", "mr", "gu", "bn"] else target_lang
        return MyMemoryTranslator(source=mm_src, target=mm_target).translate(text)
