"""
Natural Language Processing (NLP) Scene Description Module.
Turns detections into short, natural spoken guidance.

Design rules (they matter a lot for a voice that talks all day):
  * lead with what matters most: a fast-approaching object, then a close obstacle, then the rest
  * short sentences, friendly distances ("half a meter", "about 2 meters"), everyday names
  * mention a few other things in view so items held up to the camera are not ignored
  * many of one kind spread around the user are summed up ("5 cars around you")
  * only say a way past is free when nothing is close on that side
  * return stable keys (scene_key / hazard_key) so the client can avoid repeating itself

Automatic Multilanguage Communication System - SKIT Jaipur.
"""

from collections import Counter, OrderedDict, defaultdict

# Model class names -> what a person would say
SPOKEN_NAMES = {
    "dining table": "table",
    "tv": "TV screen",
    "cell phone": "phone",
    "couch": "sofa",
    "potted plant": "plant",
    "sports ball": "ball",
    "wine glass": "glass",
    "hair drier": "hair dryer",
    "motorcycle": "motorbike",
    "remote": "remote control",
    "handbag": "bag",
    "backpack": "backpack",
    "wall switch": "light switch",
    "trash can": "dustbin",
}
IRREGULAR_PLURALS = {"person": "people", "knife": "knives", "shelf": "shelves", "glasses": "glasses",
                     "keys": "keys", "scissors": "scissors", "headphones": "headphones",
                     "earphones": "earphones", "stairs": "stairs", "medicine": "medicines"}
PLURAL_ONLY = {"glasses", "keys", "scissors", "headphones", "earphones", "stairs"}
MASS_NOUNS = {"broccoli", "paper", "money", "food"}          # "some broccoli", never "a broccoli"
WHERE = {"ahead": "ahead", "left": "on your left", "right": "on your right", "around": "around you"}
SIDE_CLEAR_M = 2.0          # something on a side closer than this blocks the way past on that side
NEAREST_DISTANCE_M = 3.0    # the distance of the nearest thing is mentioned when it is this close


def spoken_name(name):
    return SPOKEN_NAMES.get(name, name)


def say_distance(m):
    """0.6 -> 'half a meter', 1.2 -> 'about 1 meter', 2.6 -> 'about 2.5 meters'."""
    if m is None:
        return ""
    if m < 0.4:
        return "very close"
    if m < 0.8:
        return "half a meter away"
    if m >= 5:
        return f"about {int(round(m))} meters away"
    half = round(m * 2) / 2
    if half == 1:
        return "about 1 meter away"
    return f"about {half:g} meters away"


class NLPDescriber:
    def __init__(self):
        self.last_description = ""
        self.last_hazard_signature = ""
        self.last_scene_key = ""

    # ------------------------------------------------------------------ wording helpers
    def _format_object_count(self, name, count):
        """'a chair', 'an apple', '2 chairs', '3 people', 'keys'."""
        name = spoken_name(name)
        if name in PLURAL_ONLY:
            return name
        if name in MASS_NOUNS:
            return f"some {name}"
        if count == 1:
            article = "an" if name[0].lower() in "aeiou" else "a"
            return f"{article} {name}"
        if name in IRREGULAR_PLURALS:
            plural = IRREGULAR_PLURALS[name]
        elif name.endswith(("s", "ch", "sh", "x")):
            plural = f"{name}es"
        elif name.endswith("y") and name[-2:-1] not in "aeiou":
            plural = f"{name[:-1]}ies"
        else:
            plural = f"{name}s"
        return f"{count} {plural}"

    @staticmethod
    def _join(items):
        items = [i for i in items if i]
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " and " + items[-1]

    def _group_by_place(self, dets, limit=5):
        """
        'a table and 2 chairs ahead, a door on your right' (nearest places first).
        Three or more of one kind in more than one place are summed up: '5 cars around you'.
        """
        totals = Counter(d["class"] for d in dets)
        sides = defaultdict(set)
        for d in dets:
            sides[d["class"]].add(d.get("spatial", "ahead"))
        spread = {c for c, n in totals.items() if n >= 3 and len(sides[c]) >= 2}
        places = OrderedDict()
        for d in dets:
            place = "around" if d["class"] in spread else d.get("spatial", "ahead")
            places.setdefault(place, []).append(d["class"])
        phrases, used = [], 0
        for place, names in places.items():
            counts = Counter(names)
            parts = []
            for name, n in counts.items():
                if used >= limit:
                    break
                parts.append(self._format_object_count(name, n))
                used += 1
            if parts:
                phrases.append(f"{self._join(parts)} {WHERE.get(place, place)}")
        if len(phrases) <= 1:
            return "".join(phrases)
        return ", ".join(phrases[:-1]) + ", and " + phrases[-1]

    def _update_text(self, full_text, others, scene_key, announced_key, must_be_full):
        """What to say for a routine update, given what the user was last told."""
        if must_be_full or announced_key is None or announced_key in ("", scene_key):
            return full_text
        before = {k for k in announced_key.split("|") if k and k != "clear"}
        if not before:
            return full_text                                   # first announcement
        if scene_key == "clear":
            return "The way ahead looks clear now."
        new = [d for d in others if f"{d['class']}:{d.get('spatial', '')}" not in before]
        now_keys = set(scene_key.split("|"))
        if not new:
            return ""                                          # things only disappeared: stay quiet
        if len(before & now_keys) == 0:
            return full_text                                   # completely different scene
        return f"Now also {self._group_by_place(new, limit=4)}."

    # ------------------------------------------------------------------ main entry
    def describe_scene(self, detections, force_generate=False, announced_key=None):
        """
        Generates a natural-language description from detected objects.
        announced_key: the scene_key the user was last told about (sent by the client). When given,
            speech_text only mentions what is new since then, so routine updates stay short.
        Returns dict: text (full description), speech_text (what to say for a routine update),
                      priority ('CRITICAL'|'HIGH'|'MEDIUM'|'NORMAL'), is_urgent, changed,
                      scene_key (what is in view), hazard_key and hazard_distance (main danger, if any).
        """
        dets = sorted(detections or [], key=lambda d: d.get("distance_m", 99))
        user = next((d for d in dets if d.get("is_user")), None)
        others = [d for d in dets if not d.get("is_user")]

        hazards = [d for d in others if d.get("severity") == "hazard"]
        cautions = [d for d in others if d.get("severity") == "caution"]
        dynamic = [d for d in others if d.get("collision_risk") == "IMMINENT_HAZARD" and d.get("ttc_seconds") is not None]

        parts = []
        priority, is_urgent = "NORMAL", False
        hazard_key, hazard_distance, lead = "", None, None

        if dynamic:
            # Something is coming at the user fast
            priority, is_urgent = "CRITICAL", True
            lead = min(dynamic, key=lambda d: d["ttc_seconds"])
            ttc = lead["ttc_seconds"]
            ttc_txt = "1 second" if ttc <= 1.4 else f"{round(ttc):g} seconds"
            action = lead.get("evasive_action", "step_right")
            move = {"step_right": "Step to your right now.", "step_left": "Step to your left now."}.get(action, "Stop now.")
            where = "straight ahead" if lead["spatial"] == "ahead" else WHERE[lead["spatial"]]
            parts.append(f"Watch out! {spoken_name(lead['class']).capitalize()} coming fast, {where}. "
                         f"Collision in about {ttc_txt}. {move}")
            hazard_key = f"dyn:{lead['class']}:{lead['spatial']}"
            hazard_distance = lead.get("distance_m")
        elif hazards:
            priority, is_urgent = "HIGH", True
            lead = hazards[0]
            name = self._format_object_count(lead["class"], 1)
            if lead["spatial"] == "ahead":
                parts.append(f"Careful, {name} right in front of you, {say_distance(lead['distance_m'])}.")
                # A side is blocked by any other hazard there, or anything close on that side
                blocked = {d["spatial"] for d in others if d is not lead and (
                    d.get("severity") == "hazard" or d.get("distance_m", 99) <= SIDE_CLEAR_M)}
                if "left" not in blocked and "right" not in blocked:
                    parts.append("You can pass on either side.")
                elif "left" not in blocked:
                    parts.append("Move to your left to pass.")
                elif "right" not in blocked:
                    parts.append("Move to your right to pass.")
                else:
                    parts.append("Stop, the way ahead is blocked.")
            else:
                parts.append(f"Careful, {name} close {WHERE[lead['spatial']]}, {say_distance(lead['distance_m'])}.")
            hazard_key = f"{lead['class']}:{lead['spatial']}"
            hazard_distance = lead.get("distance_m")
        elif cautions:
            priority = "MEDIUM"
            lead = cautions[0]
            where = "ahead of you" if lead["spatial"] == "ahead" else WHERE[lead["spatial"]]
            moving = " and coming closer" if lead.get("radial_velocity_mps", 0) > 0.3 else ""
            parts.append(f"{self._format_object_count(lead['class'], 1).capitalize()} {where}{moving}, "
                         f"{say_distance(lead['distance_m'])}.")

        # Everything else worth mentioning (the lead object is already covered)
        rest = [d for d in others if d is not lead]
        if user is not None:
            intro = "I can see you"
            if rest:
                parts.append(f"{intro}. Also in view: {self._group_by_place(rest, limit=5)}.")
            elif not parts:
                parts.append(f"{intro}. Nothing else in view.")
            else:
                parts.append(f"{intro}.")
        elif rest:
            listing = self._group_by_place(rest, limit=4 if parts else 5)
            parts.append(f"Also {listing}." if parts else f"I can see {listing}.")
            # With no warning in front, say how far the nearest thing is when it is close
            nearest = rest[0]
            if lead is None and nearest.get("distance_m", 99) <= NEAREST_DISTANCE_M:
                name = spoken_name(nearest["class"])
                which = "nearest " if sum(1 for d in rest if d["class"] == nearest["class"]) > 1 else ""
                parts.append(f"The {which}{name} is {say_distance(nearest['distance_m'])}.")

        if not parts:
            parts.append("The way ahead looks clear.")

        text = " ".join(parts).strip()

        # Stable key of what is in view: classes and places only, so distance jitter is ignored
        scene_key = "|".join(sorted({f"{d['class']}:{d.get('spatial', '')}" for d in dets})) or "clear"
        speech_text = self._update_text(text, others, scene_key, announced_key, force_generate or is_urgent)
        changed = force_generate or scene_key != self.last_scene_key or hazard_key != self.last_hazard_signature
        self.last_scene_key = scene_key
        self.last_hazard_signature = hazard_key
        self.last_description = text

        return {
            "text": text,
            "speech_text": speech_text,
            "priority": priority,
            "is_urgent": is_urgent,
            "changed": changed,
            "scene_key": scene_key,
            "hazard_key": hazard_key,
            "hazard_distance": hazard_distance,
        }
