"""
Conversational Scene Q&A Engine ("Ask AMCS").
Answers spoken or textual questions about the current surroundings for visually impaired users.

Understood questions (English, Hinglish and Hindi in Devanagari):
  * "What is in front of me?" / "Saamne kya hai?"     -> what is ahead, with distances, then the sides
  * "Describe" / "What do you see?"                   -> whole scene summary
  * "What is this?" / "What am I holding?"            -> the object held up to the camera (and its text)
  * "Who is in front of me?" / "Kaun hai?"            -> people, by name when they are saved contacts
  * "How many people / chairs are there?"             -> counting
  * "Where is the door / my bottle?"                  -> side and distance of one kind of object
  * "What is nearest?" / "What is on my left?"        -> closest object, or objects on one side
  * "Is there a place to sit?"                        -> seating
  * "What does the sign say?"                         -> text read by OCR
  * "Is the path clear?" / "Can I walk?"              -> obstacles and ground hazards
  * "Are there people / vehicles / stairs?"           -> category checks
  * "Is it dark?"                                     -> measured brightness of the picture

Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import re
import unicodedata
from collections import Counter

from modules.nlp_describer import IRREGULAR_PLURALS, PLURAL_ONLY, spoken_name

SIDE_PHRASE = {"ahead": "ahead of you", "left": "on your left", "right": "on your right"}
SEATING = {"chair", "bench", "couch", "stool", "bed"}
VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle", "auto rickshaw", "scooter", "cycle rickshaw", "train"}
TEXT_WORDS = ["sign", "board", "text", "read", "written", "say", "likha", "padho"]
SCENE_WORDS = ["what do you see", "what can you see", "describe", "surroundings", "around me", "what is around",
               "kya dikh raha hai", "aas paas"]
FRONT_WORDS = ["in front", "front of me", "ahead", "saamne", "samne", "aage"]
THIS_WORDS = ["what is this", "what is that", "what am i holding", "in my hand", "holding", "identify this",
              "ye kya hai", "yeh kya hai", "kya hai ye", "kya hai yeh", "haath mein", "pakda"]
HOLDING_WORDS = ["holding", "in my hand", "haath mein", "pakda"]
WHO_WORDS = ["who is", "who are", "who", "kaun", "anyone i know", "recognise", "recognize"]
ASK_WORDS = ["what", "anything", "something", "kya", "kuch", "tell"]
# Words a user may say for a detector class (singular form)
ALIASES = {
    "people": "person", "persons": "person", "man": "person", "woman": "person", "someone": "person",
    "somebody": "person", "anybody": "person", "anyone": "person", "human": "person",
    "phone": "cell phone", "mobile": "cell phone", "table": "dining table", "desk": "dining table",
    "sofa": "couch", "plant": "potted plant", "television": "tv", "bag": "handbag", "bike": "bicycle",
    "cycle": "bicycle", "motorbike": "motorcycle", "dustbin": "trash can", "bin": "trash can",
    "switch": "wall switch", "socket": "power socket", "key": "keys", "spectacles": "glasses",
    "specs": "glasses", "staircase": "stairs", "stair": "stairs", "steps": "stairs", "glass": "wine glass",
    "fridge": "refrigerator", "computer": "laptop", "auto": "auto rickshaw", "rickshaw": "auto rickshaw",
    "doors": "door", "darwaza": "door", "kursi": "chair", "gaadi": "car", "log": "person", "aadmi": "person",
    "aurat": "person", "insaan": "person", "botal": "bottle", "mez": "dining table", "kitab": "book",
}
STOP_WORDS = {"the", "a", "an", "is", "are", "there", "any", "my", "me", "of", "in", "on", "at", "to", "do",
              "does", "you", "see", "can", "i", "it", "what", "where", "how", "many", "much", "this", "that",
              "here", "near", "nearby", "please", "tell", "find", "show", "have", "has", "and", "or", "for",
              "front", "left", "right", "side", "ahead", "behind", "now", "room", "view",
              # Hinglish
              "kya", "hai", "hain", "ye", "yeh", "mere", "mera", "meri", "mein", "saamne", "samne", "aage", "koi",
              "kuch", "dikh", "raha", "rahi", "batao", "kahan", "kitne", "kaun", "paas", "sabse", "haath", "main",
              "hoon", "ko", "ka", "ki", "ke", "se", "par"}

# Hindi as speech recognition writes it (Devanagari) -> the Hinglish words understood above
DEVANAGARI = {
    "सामने": "saamne", "आगे": "aage", "क्या": "kya", "हैं": "hain", "है": "hai", "यह": "yeh", "ये": "ye",
    "कौन": "kaun", "कितने": "kitne", "कितनी": "kitne", "कितना": "kitne", "कहाँ": "kahan", "कहां": "kahan",
    "सबसे": "sabse", "आसपास": "aas paas", "आस पास": "aas paas", "पास": "paas", "बाएं": "left", "बाएँ": "left",
    "बाईं": "left", "बायें": "left", "बाएं तरफ": "left", "दाएं": "right", "दाएँ": "right", "दाईं": "right",
    "दायें": "right", "रास्ता": "rasta", "साफ़": "saaf", "साफ": "saaf", "सीढ़ियाँ": "seedhi", "सीढ़ियां": "seedhi",
    "सीढ़ी": "seedhi", "अंधेरा": "andhera", "रोशनी": "roshni", "पढ़ो": "padho", "पढ़िए": "padho", "लिखा": "likha",
    "बैठने": "baithne", "लोग": "log", "आदमी": "aadmi", "औरत": "aurat", "गाड़ियां": "gaadi", "गाड़ी": "gaadi",
    "दरवाज़ा": "darwaza", "दरवाजा": "darwaza", "कुर्सी": "kursi", "बोतल": "botal", "किताब": "kitab", "मेज": "mez",
    "हाथ": "haath", "में": "mein", "मेरे": "mere", "मेरा": "mera", "मेरी": "meri", "कोई": "koi", "कुछ": "kuch",
    "दिख": "dikh", "रहा": "raha", "रही": "rahi", "बताओ": "batao", "बताइए": "batao", "पकड़ा": "pakda",
    "चल": "chal", "सकता": "sakta", "सकती": "sakti", "हूँ": "hoon", "हूं": "hoon", "मैं": "main",
}
_DEVANAGARI = sorted(((unicodedata.normalize("NFC", k), v) for k, v in DEVANAGARI.items()), key=lambda kv: -len(kv[0]))
SIDE_WORDS = {"left": r"baayen|baayein|baaye|bayen|baen|baayi|baain", "right": r"daayen|daayein|daaye|dayen|daen|daayi|daain"}


def normalize_question(question):
    """Lower-case English/Hinglish with Devanagari words replaced: 'मेरे सामने क्या है' -> 'mere saamne kya hai'."""
    q = unicodedata.normalize("NFC", question or "").lower()
    for hindi, roman in _DEVANAGARI:
        q = q.replace(hindi, f" {roman} ")
    q = q.replace("what's", "what is").replace("who's", "who is").replace("where's", "where is")
    for side, pattern in SIDE_WORDS.items():
        q = re.sub(rf"\b({pattern})\b", side, q)
    q = re.sub(r"[?!.,।]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _plural(name):
    if name in PLURAL_ONLY:
        return name
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "ch", "sh", "x")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def _lower_first(text):
    """'Steps ahead' -> 'steps ahead', for use inside a sentence."""
    return text[:1].lower() + text[1:]


def _article(name):
    if name in PLURAL_ONLY:
        return name
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def _meters(d):
    m = d.get("distance_m")
    return "" if m is None else f"about {m:g} meters away"


def _place(d):
    """'on your left, about 2.1 meters away'"""
    return ", ".join(p for p in (SIDE_PHRASE.get(d.get("spatial", "ahead"), "ahead of you"), _meters(d)) if p)


def _join(items):
    """['a', 'b', 'c'] -> 'a, b and c'"""
    items = [i for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _group(dets, with_distance=True, limit=3):
    """Same-kind objects together, nearest first: ['a chair at about 1.5 meters', '2 people, the nearest at about 3 meters']."""
    groups = {}
    for d in dets:                                          # dets are sorted nearest first
        info = d.get("face_info") or {}
        name = info["name"] if info.get("recognized") else spoken_name(d.get("class", "object"))
        groups.setdefault((name, bool(info.get("recognized"))), []).append(d)
    phrases = []
    for (name, known), ds in list(groups.items())[:limit]:
        m = ds[0].get("distance_m")
        if known or len(ds) == 1:
            phrase = name if known else _article(name)
            if with_distance and m is not None:
                phrase += f" at about {m:g} meters"
        else:
            phrase = f"{len(ds)} {_plural(name)}"
            if with_distance and m is not None:
                phrase += f", the nearest at about {m:g} meters"
        phrases.append(phrase)
    return phrases


class SceneQAEngine:
    def __init__(self):
        pass

    @staticmethod
    def needs_text(question):
        """True when answering needs the (slow) text reader: sign questions, 'what is this' and descriptions."""
        q = normalize_question(question)
        return any(kw in q for kw in TEXT_WORDS + SCENE_WORDS + THIS_WORDS) or SceneQAEngine._is_front(q)

    @staticmethod
    def needs_faces(question):
        """True for 'who is in front of me' style questions (face recognition runs only for these)."""
        return SceneQAEngine._has(normalize_question(question), WHO_WORDS)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _has(q, keywords):
        """Whole-word match for single words, substring match for phrases."""
        words = set(re.findall(r"[a-z']+", q))
        return any((kw in q) if " " in kw else (kw in words) for kw in keywords)

    @staticmethod
    def _is_front(q):
        """'What is in front of me?', 'what is ahead', 'saamne kya hai' (but not 'is there a chair ahead')."""
        return SceneQAEngine._has(q, FRONT_WORDS) and SceneQAEngine._has(q, ASK_WORDS)

    @staticmethod
    def _target_classes(q, detections):
        """Detector classes the question is about, e.g. 'bottles' -> {'bottle'}."""
        present = {d.get("class", "").lower() for d in detections}
        targets = set()
        for word in re.findall(r"[a-z]+", q):
            if word in STOP_WORDS:
                continue
            forms = {word, word.rstrip("s"), word[:-2] if word.endswith("es") else word}
            for form in forms:
                cls = ALIASES.get(form, form)
                if cls in present:
                    targets.add(cls)
                    continue
                # 'bottle' matches 'water bottle'; 'table' matches 'dining table'
                targets.update(c for c in present if len(form) >= 3 and form in c.split())
        return targets

    @staticmethod
    def _asked_noun(q):
        """Best guess at the thing asked about when nothing in view matches ('where is my wallet')."""
        words = [w for w in re.findall(r"[a-z]+", q) if w not in STOP_WORDS]
        return words[-1] if words else ""

    # ------------------------------------------------------------------ main entry
    def answer_question(self, question, detections=None, ocr_items=None, ground_hazards=None, ocr_text="",
                        lang="en", brightness=None):
        """
        Answers a user's visual question based on real-time scene detections, OCR, and spatial layout.
        brightness: mean brightness of the camera picture (0-255), if a picture was sent.
        Returns a concise, natural, spoken answer.
        """
        if not question or not question.strip():
            return "Please ask a question about your surroundings."

        q = normalize_question(question)
        ocr_items = ocr_items or []
        ground_hazards = ground_hazards or []
        # The user themself (front camera) is not part of the surroundings
        detections = sorted((d for d in (detections or []) if not d.get("is_user")),
                            key=lambda d: d.get("distance_m") if d.get("distance_m") is not None else 99)
        if not ocr_text and ocr_items:
            ocr_text = " ".join(item.get("text", "") for item in ocr_items if item.get("text"))

        # 1. "Who is in front of me?"
        if self._has(q, WHO_WORDS):
            return self._who(detections)

        # 2. "What is this?" / "What am I holding?"
        if any(kw in q for kw in THIS_WORDS):
            return self._identify_this(detections, ocr_text, holding=any(kw in q for kw in HOLDING_WORDS))

        # 3. "How many chairs are there?"
        if "how many" in q or "count" in q or self._has(q, ["kitne"]):
            return self._count(q, detections)

        # 4. "What is nearest / closest to me?"
        if self._has(q, ["nearest", "closest"]) or "sabse paas" in q:
            if not detections:
                return "I do not see anything near you right now."
            d = detections[0]
            return f"The nearest thing is {_article(spoken_name(d['class']))}, {_place(d)}."

        # 5. "What is on my left / right?"
        for side in ("left", "right"):
            if self._has(q, [side]) and self._has(q, ASK_WORDS):
                on_side = [d for d in detections if d.get("spatial") == side]
                if not on_side:
                    return f"I do not see anything on your {side}."
                return f"On your {side}: {_join(_group(on_side, limit=4))}."

        # "Where is the bus / the chair?" asks for a position, whatever category the object is in
        if self._has(q, ["where", "kahan"]):
            located = self._locate(q, detections)
            if located:
                return located

        # 6. "Is the path clear?" / "Can I walk?" / "Any obstacles?"
        if any(kw in q for kw in ["path clear", "way clear", "can i walk", "can i go", "safe to walk", "obstacle",
                                  "rasta saaf", "chal sakta", "chal sakti"]):
            return self._path(detections, ground_hazards)

        # 7. "What does the sign say?" / "Read the board / text"
        if self._has(q, TEXT_WORDS):
            if ocr_text and ocr_text.strip():
                return f"The sign reads: {ocr_text.strip()}."
            return "No readable text or signboards are visible in the camera view."

        # 8. "What is in front of me?" / "Saamne kya hai?"
        if self._is_front(q):
            return self._describe_front(detections, ocr_text, ground_hazards)

        # 9. "What do you see?" / "Describe scene"
        if any(kw in q for kw in SCENE_WORDS):
            return self._describe_comprehensive_scene(detections, ocr_text, ground_hazards)

        # 10. "Is there a place to sit?" / "Find a chair / bench"
        if self._has(q, ["sit", "seat", "seating", "chair", "bench", "couch", "sofa", "baithne", "kursi"]):
            seating = [d for d in detections if d.get("class") in SEATING]
            if seating:
                nearest = seating[0]
                return f"Yes, there is {_article(spoken_name(nearest['class']))} {_place(nearest)}."
            return "No chairs or benches are currently visible in front of you."

        # 11. "Are there vehicles or traffic?" / "Any car?"
        if self._has(q, ["vehicle", "vehicles", "car", "cars", "bike", "motorcycle", "bus", "truck", "traffic", "gaadi"]):
            vehicles = [d for d in detections if d.get("class") in VEHICLES]
            if vehicles:
                nearest = vehicles[0]
                count = "1 vehicle" if len(vehicles) == 1 else f"{len(vehicles)} vehicles"
                return (f"Careful: {count} in view. The nearest is {_article(spoken_name(nearest['class']))} "
                        f"{_place(nearest)}.")
            return "No vehicles detected in your immediate path."

        # 12. "Are there stairs or curbs?" / "Watch step"
        if self._has(q, ["stair", "stairs", "curb", "step", "steps", "pothole", "seedhi"]):
            stairs = [d for d in detections if d.get("class") == "stairs"]
            if stairs:
                return f"Yes, stairs {_place(stairs[0])}. Take care."
            if ground_hazards:
                return f"Yes, {_lower_first(ground_hazards[0].get('description', ''))}."
            return "No stairs or ground drop-offs detected immediately ahead."

        # 13. "Is it dark or bright?"
        if self._has(q, ["dark", "light", "bright", "lighting", "andhera", "roshni"]) and not self._target_classes(q, detections):
            if brightness is None:
                return "I need a camera picture to judge the lighting."
            if brightness < 45:
                return "It is quite dark here. Turning on a light or the torch will help me see better."
            if brightness < 90:
                return "The light is dim, but I can still see."
            return "It is bright here. Lighting is good for assistive vision."

        # 14. "Where is the door?" / "Is there a bottle?" / "Are there people?"
        located = self._locate(q, detections)
        if located:
            return located
        if self._has(q, ["person", "people", "someone", "somebody", "anybody", "anyone", "koi", "log"]):
            return "No persons detected in front of you."
        if self._has(q, ["where", "find", "kahan"]) or q.startswith(("is there", "are there", "do you see", "can you see")):
            noun = self._asked_noun(q)
            if noun:
                return f"I do not see {_article(noun)} in view right now. Try turning the camera slowly."

        # Not understood: say so, then give the most useful answer, what is ahead
        return f"I am not sure what you asked, so here is what I see. {self._describe_front(detections, ocr_text, ground_hazards)}"

    # ------------------------------------------------------------------ answers
    def _locate(self, q, detections):
        """'Yes, I see a door on your left, about 4 meters away.' or None if it is not in view."""
        targets = self._target_classes(q, detections)
        if not targets:
            return None
        matching = [d for d in detections if d.get("class", "").lower() in targets]
        nearest = matching[0]
        name = spoken_name(nearest["class"])
        if len(matching) == 1:
            return f"Yes, I see {_article(name)} {_place(nearest)}."
        return f"Yes, I see {len(matching)} {_plural(name)}. The nearest is {_place(nearest)}."

    def _count(self, q, detections):
        targets = self._target_classes(q, detections)
        if targets:
            parts = []
            for cls in sorted(targets):
                n = sum(1 for d in detections if d.get("class", "").lower() == cls)
                name = spoken_name(cls)
                parts.append(name if name in PLURAL_ONLY else (f"1 {name}" if n == 1 else f"{n} {_plural(name)}"))
            return f"I can see {' and '.join(parts)}."
        noun = self._asked_noun(q.replace("how many", "").replace("count", ""))
        if noun and noun not in ("things", "objects", "items", "thing", "object"):
            return f"I do not see any {noun} right now."
        if not detections:
            return "I do not see any objects right now."
        counts = Counter(spoken_name(d["class"]) for d in detections)
        listing = ", ".join(n if n in PLURAL_ONLY else (f"1 {n}" if c == 1 else f"{c} {_plural(n)}")
                            for n, c in counts.most_common(5))
        return f"I can see {len(detections)} objects: {listing}."

    def _path(self, detections, ground_hazards):
        hazards = [d for d in detections if d.get("severity") == "hazard"]
        if hazards:
            h = hazards[0]
            count = "is 1 hazard" if len(hazards) == 1 else f"are {len(hazards)} hazards"
            return (f"Caution. There {count} close to you. {_article(spoken_name(h.get('class', 'obstacle'))).capitalize()} "
                    f"{_place(h)}.")
        if ground_hazards:
            return f"Watch your step. {ground_hazards[0].get('description')}."
        obstacles = [d for d in detections if d.get("is_obstacle", True) and d.get("spatial", "ahead") == "ahead"]
        if obstacles:
            o = obstacles[0]
            return f"Path is mostly clear. The nearest obstacle ahead is {_article(spoken_name(o.get('class')))}, {_meters(o)}."
        return "The path ahead is clear. You can walk safely."

    def _who(self, detections):
        """'Rahul, your friend, is ahead of you, about 2 meters away. 1 more person on your left.'"""
        people = [d for d in detections if d.get("class") == "person" or d.get("face_info")]
        if not people:
            return "I do not see anyone in front of you."
        known = [d for d in people if (d.get("face_info") or {}).get("recognized")]
        parts = []
        for d in known[:3]:
            info = d["face_info"]
            role = f", your {info['role'].lower()}," if info.get("role") else ""
            parts.append(f"{info['name']}{role} is {_place(d)}.")
        others = [d for d in people if d not in known]
        if others:
            if known:
                who = "a person" if len(others) == 1 else f"{len(others)} people"
                nearest = "" if len(others) == 1 else "the nearest "
                parts.append(f"I also see {who} I do not recognise, {nearest}{_place(others[0])}.")
            elif len(others) == 1:
                parts.append(f"There is a person {_place(others[0])}. I do not recognise them.")
            else:
                parts.append(f"There are {len(others)} people. The nearest is {_place(others[0])}. I do not recognise them.")
        return " ".join(parts)

    def _identify_this(self, detections, ocr_text, holding=False):
        """The object held up to the camera: the nearest thing in the middle of the view."""
        things = [d for d in detections if not (holding and d.get("class") == "person")]
        centred = [d for d in things if d.get("spatial", "ahead") == "ahead"]
        text = (ocr_text or "").strip()[:80]
        said = f" The text on it says: {text}." if text else ""
        if not (centred or things):
            if text:
                return f"I cannot tell what it is, but the text on it says: {text}."
            return ("I cannot recognise it. Hold it about 30 centimeters in front of the camera, in good light, "
                    "and ask again.")
        d = (centred or things)[0]
        name = _article(spoken_name(d["class"]))
        guess = f"It might be {name}" if (d.get("confidence") or 1.0) < 0.5 else f"It looks like {name}"
        where = "" if holding or d.get("spatial", "ahead") == "ahead" else f", {SIDE_PHRASE.get(d.get('spatial'))}"
        return f"{guess}{where}.{said}"

    def _describe_front(self, detections, ocr_text, ground_hazards):
        """What is ahead first, with distances; then each side briefly; then a sign, if any."""
        parts = []
        if ground_hazards:
            parts.append(f"Careful: {_lower_first(ground_hazards[0]['description'])}.")
        ahead = [d for d in detections if d.get("spatial", "ahead") == "ahead"]
        if ahead:
            lead = "Careful. Ahead of you" if ahead[0].get("severity") == "hazard" else "Ahead of you"
            parts.append(f"{lead}: {_join(_group(ahead))}.")
        elif detections:
            parts.append("Nothing is directly in front of you.")
        else:
            parts.append("I do not see any objects in front of you.")
        for side in ("left", "right"):
            on_side = [d for d in detections if d.get("spatial") == side]
            if on_side:
                parts.append(f"On your {side}: {_join(_group(on_side, with_distance=False, limit=2))}.")
        if ocr_text and ocr_text.strip():
            parts.append(f"A sign reads: {ocr_text.strip()[:60]}.")
        return " ".join(parts)

    def _describe_comprehensive_scene(self, detections, ocr_text, ground_hazards):
        """Builds a rich, cohesive narrative of the entire scene."""
        parts = []

        if ground_hazards:
            parts.append(f"At ground level: {_lower_first(ground_hazards[0]['description'])}.")

        if detections:
            desc_parts = []
            for side, phrase in (("ahead", "directly ahead"), ("left", "on your left"), ("right", "on your right")):
                counts = Counter(spoken_name(d["class"]) for d in detections if d.get("spatial", "ahead") == side)
                if counts:
                    names = [_article(n) if c == 1 else f"{c} {_plural(n)}" for n, c in list(counts.items())[:3]]
                    desc_parts.append(f"{', '.join(names)} {phrase}")
            nearest = detections[0]
            parts.append(f"I can see {'; '.join(desc_parts)}.")
            if nearest.get("distance_m") is not None:
                parts.append(f"The nearest is the {spoken_name(nearest['class'])}, {_meters(nearest)}.")
        else:
            parts.append("I do not see any objects around you.")

        if ocr_text and ocr_text.strip():
            clean = ocr_text.strip()[:60]
            parts.append(f"Signboard text detected: '{clean}'.")

        return " ".join(parts)
