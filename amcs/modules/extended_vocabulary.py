"""
Extended object vocabulary for the open-vocabulary YOLOE detector.

YOLO26 (the main detector) knows the 80 COCO classes (people, vehicles, furniture, cups,
bottles, phones...). This list adds everyday things a visually impaired person needs that COCO
does not cover: doors, stairs, switches, cabinets, keys, wallets, pens, glasses and so on.

Each entry: name -> (typical height in metres, is_obstacle)
  - the height feeds monocular distance estimation
  - only obstacles can raise "careful" proximity warnings; small items are just described

The vocabulary is baked into `yoloe-26s-amcs.pt` by tools/build_extended_model.py, so the app
does not need the (large) text encoder at runtime. Re-run that script after editing this list.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

EXTENDED_VOCABULARY = {
    # Building structure & navigation
    "door": (2.00, True),
    "stairs": (1.00, True),
    "window": (1.20, False),
    "wall switch": (0.10, False),
    "power socket": (0.08, False),
    "pillar": (2.50, True),
    "railing": (1.00, True),
    "gate": (1.80, True),
    "elevator": (2.10, True),
    "escalator": (1.50, True),
    "trash can": (0.70, True),
    "traffic cone": (0.70, True),
    "pole": (3.00, True),
    "auto rickshaw": (1.70, True),
    "scooter": (1.10, True),
    "cycle rickshaw": (1.70, True),
    "handcart": (1.00, True),
    "speed bump": (0.10, False),
    "manhole": (0.05, False),
    # Furniture & fittings
    "cabinet": (0.90, True),
    "shelf": (1.50, True),
    "wardrobe": (1.90, True),
    "desk": (0.75, True),
    "stool": (0.45, True),
    "curtain": (2.00, False),
    "mirror": (0.80, False),
    "ceiling fan": (0.30, False),
    "table fan": (0.50, False),
    "air conditioner": (0.30, False),
    "lamp": (0.50, False),
    "water cooler": (1.30, True),
    "bucket": (0.35, True),
    "box": (0.40, True),
    # Kitchen
    "kettle": (0.25, False),
    "stove": (0.15, False),
    "gas cylinder": (0.75, True),
    "pressure cooker": (0.30, False),
    "pan": (0.10, False),
    "plate": (0.03, False),
    "tap": (0.15, False),
    # Personal items
    "keys": (0.06, False),
    "wallet": (0.10, False),
    "pen": (0.14, False),
    "glasses": (0.05, False),
    "watch": (0.04, False),
    "headphones": (0.18, False),
    "earphones": (0.03, False),
    "charger": (0.08, False),
    "medicine": (0.10, False),
    "shoe": (0.12, False),
    "slipper": (0.06, False),
    "cap": (0.12, False),
    "helmet": (0.25, False),
    "pillow": (0.15, False),
    "blanket": (0.30, False),
    "towel": (0.40, False),
    "bag": (0.40, False),
    "paper": (0.28, False),
    "banknote": (0.07, False),
    "coin": (0.02, False),
}

EXTENDED_CLASS_NAMES = list(EXTENDED_VOCABULARY.keys())
EXTENDED_HEIGHTS = {k: v[0] for k, v in EXTENDED_VOCABULARY.items()}
EXTENDED_OBSTACLES = {k for k, v in EXTENDED_VOCABULARY.items() if v[1]}
