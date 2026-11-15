"""
Medicine Blister Strip & Expiry Date Reader Module.
Empowers visually impaired individuals to safely identify medications, dosages,
and check whether a blister strip or bottle has expired.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np
import re
from datetime import datetime

# Common medicinal drug database (Generic names and popular Indian/global brands)
MEDICINE_DATABASE = {
    # Antipyretic & Analgesic (Fever / Pain)
    "dolo": {"generic": "Paracetamol", "category": "Pain / Fever", "hi": "डोलो 650 (बुखार और दर्द निवारक)"},
    "paracetamol": {"generic": "Paracetamol", "category": "Pain / Fever", "hi": "पैरासिटामोल (बुखार की दवा)"},
    "crocin": {"generic": "Paracetamol", "category": "Pain / Fever", "hi": "क्रोसिन (बुखार की दवा)"},
    "calpol": {"generic": "Paracetamol", "category": "Pain / Fever", "hi": "कैलपोल (बुखार की दवा)"},
    "combiflam": {"generic": "Ibuprofen + Paracetamol", "category": "Pain Relief", "hi": "कॉम्बीफ्लेम (दर्द निवारक)"},
    "saridon": {"generic": "Propyphenazone + Paracetamol", "category": "Headache Relief", "hi": "सैरिडॉन (सिरदर्द की दवा)"},
    "aspirin": {"generic": "Acetylsalicylic Acid", "category": "Blood Thinner / Pain", "hi": "एस्पिरिन"},
    "disprin": {"generic": "Aspirin Soluble", "category": "Pain / Headache", "hi": "डिस्प्रिन"},
    "ibuprofen": {"generic": "Ibuprofen", "category": "NSAID Pain Relief", "hi": "आइबुप्रोफेन (सूजन व दर्द निवारक)"},
    "diclofenac": {"generic": "Diclofenac Sodium", "category": "NSAID Pain Relief", "hi": "डाइक्लोफेनाक (जोड़ों का दर्द)"},
    "voveran": {"generic": "Diclofenac", "category": "Pain Relief", "hi": "वोवेरान (दर्द निवारक)"},

    # Antibiotics
    "azithromycin": {"generic": "Azithromycin", "category": "Antibiotic", "hi": "एज़िथ्रोमाइसिन (एंटीबायोटिक)"},
    "amoxicillin": {"generic": "Amoxicillin", "category": "Antibiotic", "hi": "अमोक्सीसिलिन (एंटीबायोटिक)"},
    "augmentin": {"generic": "Amoxicillin + Clavulanic", "category": "Antibiotic", "hi": "ऑगमेंटिन (एंटीबायोटिक)"},
    "ciprofloxacin": {"generic": "Ciprofloxacin", "category": "Antibiotic", "hi": "सिप्रोफ्लोक्सासिन (एंटीबायोटिक)"},
    "ofloxacin": {"generic": "Ofloxacin", "category": "Antibiotic", "hi": "ओफ्लोक्सासिन (एंटीबायोटिक)"},
    "cefixime": {"generic": "Cefixime", "category": "Antibiotic", "hi": "सेफिक्सिम (एंटीबायोटिक)"},

    # Antihistamines & Allergy
    "cetirizine": {"generic": "Cetirizine", "category": "Antiallergic", "hi": "सिट्रीज़िन (एलर्जी और जुकाम की दवा)"},
    "levocetirizine": {"generic": "Levocetirizine", "category": "Antiallergic", "hi": "लेवोसिट्रीज़िन (एलर्जी की दवा)"},
    "allegra": {"generic": "Fexofenadine", "category": "Antiallergic", "hi": "एलेग्रा (एलर्जी निवारक)"},
    "montair": {"generic": "Montelukast", "category": "Asthma / Allergy", "hi": "मोंटेयर एलसी (एलर्जी व सांस)"},
    "montelukast": {"generic": "Montelukast", "category": "Asthma / Allergy", "hi": "मोंटेलुकास्ट (सांस व एलर्जी)"},

    # Gastrointestinal (Acidity / Gas)
    "pantoprazole": {"generic": "Pantoprazole", "category": "Antacid / Acid Reflux", "hi": "पैंटोप्रैज़ोल (गैस और एसिडिटी)"},
    "pan": {"generic": "Pantoprazole", "category": "Antacid", "hi": "पैन 40 (एसिडिटी की दवा)"},
    "pantocid": {"generic": "Pantoprazole", "category": "Antacid", "hi": "पैन्टोसिड (एसिडिटी की दवा)"},
    "omeprazole": {"generic": "Omeprazole", "category": "Antacid", "hi": "ओमेप्राजोल (पेट की गैस)"},
    "omez": {"generic": "Omeprazole", "category": "Antacid", "hi": "ओमेज़ (एसिडिटी की दवा)"},
    "rabeprazole": {"generic": "Rabeprazole", "category": "Antacid", "hi": "रेबेप्राजोल (एसिडिटी की दवा)"},
    "ranitidine": {"generic": "Ranitidine", "category": "Antacid", "hi": "रैनिटिडिन (गैस की दवा)"},
    "digene": {"generic": "Magnesium / Aluminium Hydroxide", "category": "Antacid", "hi": "डाइजीन (पेट की जलन व गैस)"},

    # Blood Pressure & Heart
    "telmisartan": {"generic": "Telmisartan", "category": "Blood Pressure", "hi": "टेल्मिसार्टन (ब्लड प्रेशर नियंत्रण)"},
    "telma": {"generic": "Telmisartan", "category": "Blood Pressure", "hi": "टेल्मा (हाई बीपी की दवा)"},
    "amlodipine": {"generic": "Amlodipine", "category": "Blood Pressure", "hi": "एम्लोडिपिन (ब्लड प्रेशर)"},
    "atenolol": {"generic": "Atenolol", "category": "Beta Blocker", "hi": "एटेनोलोल (हृदय व बीपी)"},
    "atorvastatin": {"generic": "Atorvastatin", "category": "Cholesterol", "hi": "एटोरवास्टेटिन (कोलेस्ट्रॉल)"},

    # Diabetes
    "metformin": {"generic": "Metformin", "category": "Diabetes", "hi": "मेटफॉर्मिन (शुगर / डायबिटीज)"},
    "glycomet": {"generic": "Metformin", "category": "Diabetes", "hi": "ग्लाइकोमेट (डायबिटीज की दवा)"},
    "glimepiride": {"generic": "Glimepiride", "category": "Diabetes", "hi": "ग्लिमेपिराइड (डायबिटीज)"},

    # Vitamins & Supplements
    "neurobion": {"generic": "Vitamin B Complex + B12", "category": "Multivitamin", "hi": "न्यूरोबियन (विटामिन बी व नसों के लिए)"},
    "becosules": {"generic": "Vitamin B Complex + C", "category": "Multivitamin", "hi": "बिकासूल्स (मुंह के छाले व कमजोरी)"},
    "zincovit": {"generic": "Multivitamin + Zinc", "category": "Nutritional Supplement", "hi": "जिंकोविट (इम्युनिटी व विटामिन)"},
    "limcee": {"generic": "Vitamin C 500mg", "category": "Vitamin Supplement", "hi": "लिम्सी (विटामिन सी चबाने योग्य)"}
}

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

MONTH_NAMES_EN = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTH_NAMES_HI = ["", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


class MedicineReader:
    """
    Dedicated AI analyzer for medicinal blister strips and pharmacy labels.
    Extracts medicine name, strength/dosage, and validates expiry dates.
    """

    def __init__(self):
        # Current reference date
        self.now = datetime.now()

    def preprocess_foil(self, frame):
        """
        Enhances contrast and reduces metallic specular glare from aluminum blister packs
        using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        if frame is None or frame.size == 0:
            return None

        # Convert to Grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # CLAHE enhances micro-contrast of stamped/embossed letters on foil
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter reduces foil texture noise while keeping font edges crisp
        denoised = cv2.bilateralFilter(enhanced, 7, 75, 75)

        # Convert back to 3-channel BGR representation for seamless pipeline compatibility
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def parse_medicine_info(self, raw_text):
        """
        Extracts medicine name, dosage, and expiry information from OCR text.
        Returns parsed metadata dictionary.
        """
        if not raw_text or not isinstance(raw_text, str):
            return {
                "detected": False,
                "medicine_name": None,
                "dosage": None,
                "full_name": None,
                "expiry_str": None,
                "expiry_month": None,
                "expiry_year": None,
                "is_expired": None,
                "months_difference": None,
                "category": None,
                "raw_text": ""
            }

        text_lower = raw_text.lower()
        clean_text = re.sub(r'[\r\n\t]+', ' ', raw_text)

        # 1. Match Medicine Name
        matched_key = None
        matched_meta = None

        # Sort keys by length descending to match compound words first
        for key in sorted(MEDICINE_DATABASE.keys(), key=len, reverse=True):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, text_lower):
                matched_key = key
                matched_meta = MEDICINE_DATABASE[key]
                break

        # If no strict word boundary match, check substring for names >= 5 chars
        if not matched_key:
            for key in sorted(MEDICINE_DATABASE.keys(), key=len, reverse=True):
                if len(key) >= 5 and key in text_lower:
                    matched_key = key
                    matched_meta = MEDICINE_DATABASE[key]
                    break

        medicine_name = matched_key.title() if matched_key else None
        category = matched_meta["category"] if matched_meta else None

        # 2. Extract Strength / Dosage (e.g. 650mg, 500 mg, 10mg, 40 mg)
        dosage = None
        dosage_match = re.search(r'\b(\d{1,4}(?:\.\d{1,2})?)\s*(mg|mcg|gm?|ml|iu)\b', text_lower)
        if dosage_match:
            dosage = f"{dosage_match.group(1)}{dosage_match.group(2)}"
        elif "650" in text_lower:
            dosage = "650mg"
        elif "500" in text_lower:
            dosage = "500mg"
        elif "40" in text_lower and matched_key in ["pan", "pantoprazole", "omeprazole"]:
            dosage = "40mg"

        # Full Display Name
        if medicine_name and dosage:
            full_name = f"{medicine_name} {dosage}"
        elif medicine_name:
            full_name = medicine_name
        else:
            full_name = None

        # 3. Extract Expiry Date
        expiry_info = self._extract_expiry(raw_text)

        # Calculate validity / expiration
        is_expired = None
        months_diff = None
        if expiry_info["month"] and expiry_info["year"]:
            is_expired, months_diff = self.check_expiry(expiry_info["month"], expiry_info["year"])

        detected = bool(medicine_name or expiry_info["expiry_str"])

        return {
            "detected": detected,
            "medicine_name": medicine_name,
            "dosage": dosage,
            "full_name": full_name,
            "expiry_str": expiry_info["expiry_str"],
            "expiry_month": expiry_info["month"],
            "expiry_year": expiry_info["year"],
            "is_expired": is_expired,
            "months_difference": months_diff,
            "category": category,
            "meta": matched_meta,
            "raw_text": clean_text
        }

    def _extract_expiry(self, text):
        """
        Scans text for expiry patterns (e.g., 'EXP: 08/2027', 'EXP. 12/26', 'USE BEFORE 03-2028', 'EXP OCT 2026').
        """
        # Normalize text
        normalized = text.replace('\n', ' ')

        # Pattern A: EXP / EXPIRY / USE BEFORE / BB followed by Month and Year numbers (MM/YY or MM/YYYY)
        # e.g., "EXP: 08/2027", "EXP. 05/26", "USE BEFORE 11/2025"
        pattern_num = re.search(
            r'(?:EXP(?:IRY)?|USE\s*BEFORE|BEST\s*BEFORE|BB|B\.?B\.?)[\.\s:]*([0-1]?[0-9])[\/\.\-\s](20\d{2}|\d{2})\b',
            normalized,
            re.IGNORECASE
        )
        if pattern_num:
            month = int(pattern_num.group(1))
            year = int(pattern_num.group(2))
            if 1 <= month <= 12:
                year = year if year >= 2000 else 2000 + year
                return {"expiry_str": f"{month:02d}/{year}", "month": month, "year": year}

        # Pattern B: EXP with month name (e.g., "EXP: AUG 2027", "EXPIRY: NOV 26", "EXP OCT/2026")
        pattern_text = re.search(
            r'(?:EXP(?:IRY)?|USE\s*BEFORE|BB)[\.\s:]*([A-Za-z]{3,9})[\.\s\/\-\']+(20\d{2}|\d{2})\b',
            normalized,
            re.IGNORECASE
        )
        if pattern_text:
            m_str = pattern_text.group(1).lower()
            year = int(pattern_text.group(2))
            year = year if year >= 2000 else 2000 + year
            for k, v in MONTH_MAP.items():
                if m_str.startswith(k):
                    return {"expiry_str": f"{v:02d}/{year}", "month": v, "year": year}

        # Pattern C: Generic date format MM/YYYY or MM/YY appearing near the word EXP
        pattern_exp_near = re.search(
            r'\bEXP\b.*?([0-1]?[0-9])[\/\-](20\d{2}|\d{2})\b',
            normalized,
            re.IGNORECASE
        )
        if pattern_exp_near:
            month = int(pattern_exp_near.group(1))
            year = int(pattern_exp_near.group(2))
            if 1 <= month <= 12:
                year = year if year >= 2000 else 2000 + year
                return {"expiry_str": f"{month:02d}/{year}", "month": month, "year": year}

        # Pattern D: Standalone Month-Year token like 08/2028 or 04/2025
        pattern_standalone = re.search(r'\b(0[1-9]|1[0-2])[\/](202[4-9]|203[0-5])\b', normalized)
        if pattern_standalone:
            month = int(pattern_standalone.group(1))
            year = int(pattern_standalone.group(2))
            return {"expiry_str": f"{month:02d}/{year}", "month": month, "year": year}

        return {"expiry_str": None, "month": None, "year": None}

    def check_expiry(self, exp_month, exp_year):
        """
        Compares expiry month/year with system date.
        Returns:
            is_expired: bool (True if expired, False if still valid)
            months_diff: int (positive = months remaining, negative = months past expired)
        """
        cur_year = self.now.year
        cur_month = self.now.month

        # Calculate difference in months: (exp_year - cur_year)*12 + (exp_month - cur_month)
        months_diff = (exp_year - cur_year) * 12 + (exp_month - cur_month)

        # A medicine is valid through the end of its expiry month
        is_expired = months_diff < 0
        return is_expired, months_diff

    def generate_advisory(self, parsed_info):
        """
        Generates spoken auditory guidance in English and Hindi.
        """
        if not parsed_info["detected"]:
            return {
                "advisory_en": "No medicine label or expiry date clearly visible. Please rotate the blister strip towards the light.",
                "advisory_hi": "कोई दवाई या एक्सपायरी तारीख स्पष्ट नहीं दिखी। कृपया पत्ते को रोशनी में घुमाएं।"
            }

        full_name = parsed_info["full_name"]
        is_expired = parsed_info["is_expired"]
        month = parsed_info["expiry_month"]
        year = parsed_info["expiry_year"]
        meta = parsed_info.get("meta")

        # Expiry text representation
        date_str_en = ""
        date_str_hi = ""
        if month and year and 1 <= month <= 12:
            m_en = MONTH_NAMES_EN[month]
            m_hi = MONTH_NAMES_HI[month]
            date_str_en = f"{m_en} {year}"
            date_str_hi = f"{m_hi} {year}"

        # 1. Expired Medicine Alert (Critical Hazard!)
        if is_expired:
            if full_name:
                en = f"DANGER! {full_name} is EXPIRED! Expired in {date_str_en}. Do not consume!"
                hi_name = meta["hi"] if meta else full_name
                hi = f"सावधान! {hi_name} की अवधि समाप्त हो चुकी है! यह {date_str_hi} में एक्सपायर हो गई। इसे न खाएं!"
            else:
                en = f"DANGER! This medicine is EXPIRED! Expired in {date_str_en}. Do not take this medication!"
                hi = f"सावधान! यह दवाई एक्सपायर हो चुकी है! एक्सपायरी {date_str_hi} थी। कृपया इसे न लें!"

        # 2. Valid Medicine Reassurance
        elif is_expired is False:
            if full_name:
                category = parsed_info["category"]
                cat_desc = f" ({category})" if category else ""
                en = f"Identified {full_name}{cat_desc}. Safe to use. Valid until {date_str_en}."
                hi_name = meta["hi"] if meta else full_name
                hi = f"पहचानी गई: {hi_name}। सुरक्षित है, वैधता {date_str_hi} तक है।"
            else:
                en = f"Medicine strip is safe and valid until {date_str_en}."
                hi = f"यह दवाई सुरक्षित है, वैधता {date_str_hi} तक मान्य है।"

        # 3. Only Medicine Identified (no expiry found)
        elif full_name:
            category = parsed_info["category"]
            cat_desc = f", used for {category}" if category else ""
            en = f"Identified {full_name}{cat_desc}. Expiry date not visible, check other side."
            hi_name = meta["hi"] if meta else full_name
            hi = f"पहचानी गई: {hi_name}। एक्सपायरी तारीख नहीं दिखी, कृपया दूसरी तरफ दिखाएं।"

        else:
            en = "Medicine text detected, but name not confirmed."
            hi = "दवाई का विवरण मिला, पर नाम की पुष्टि नहीं हुई।"

        return {"advisory_en": en, "advisory_hi": hi}

    def analyze_strip(self, frame=None, raw_text=""):
        """
        High-level analysis method.
        Accepts optional image frame and OCR extracted text.
        Returns complete analysis payload with speech advisories.
        """
        parsed = self.parse_medicine_info(raw_text)
        advisories = self.generate_advisory(parsed)

        parsed["advisory_en"] = advisories["advisory_en"]
        parsed["advisory_hi"] = advisories["advisory_hi"]
        return parsed
