"""Turns raw detections into a short, natural-language scene description.

Rule-based NLP: group by object, pluralise, order by urgency and phrase
direction/distance the way a person would say it.
"""
from collections import defaultdict

_URGENCY = {"very close": 0, "close": 1, "far": 2}
_IRREGULAR_PLURALS = {
    "person": "people",
    "bus": "buses",
    "knife": "knives",
    "sheep": "sheep",
    "scissors": "scissors",
    "skis": "skis",
    "mouse": "mice",
    "tv": "TVs",
}
_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def _plural(label):
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith(("s", "x", "ch", "sh")):
        return label + "es"
    return label + "s"


def _place(direction):
    return "ahead" if direction == "ahead" else f"on your {direction}"


def _phrase(label, items):
    """e.g. 'a chair on your left' / 'two people ahead'."""
    directions = {d["direction"] for d in items}
    count = len(items)
    noun = f"{_article(label)} {label}" if count == 1 else f"{_NUMBERS.get(count, count)} {_plural(label)}"
    where = _place(next(iter(directions))) if len(directions) == 1 else "around you"
    return f"{noun} {where}"


def describe(detections):
    """Return {'text': str, 'alerts': [str]} for a list of detection dicts."""
    if not detections:
        return {"text": "I can't see anything clearly right now.", "alerts": []}

    # Safety first: anything very close and in the walking path.
    alerts = []
    for d in sorted(detections, key=lambda d: _URGENCY[d["distance"]]):
        if d["distance"] == "very close":
            alerts.append(f"Caution, {_article(d['label'])} {d['label']} is very close, {_place(d['direction'])}.")

    groups = defaultdict(list)
    for d in detections:
        groups[d["label"]].append(d)

    # Order groups by their nearest member.
    ordered = sorted(groups.items(), key=lambda kv: min(_URGENCY[i["distance"]] for i in kv[1]))
    phrases = [_phrase(label, items) for label, items in ordered]

    if len(phrases) == 1:
        listing = phrases[0]
    else:
        listing = ", ".join(phrases[:-1]) + " and " + phrases[-1]

    sentence = f"I can see {listing}."
    text = " ".join(alerts + [sentence])
    return {"text": text, "alerts": alerts}
