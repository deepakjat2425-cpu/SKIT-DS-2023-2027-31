"""
Face & Social Interaction Engine for Visually Impaired Users.
Enables known person identification ("Who is in front of me?") and social cue / expression awareness.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import config

REGISTRY_FILE = config.BASE_DIR / "registered_contacts.json"

class FaceSocialEngine:
    def __init__(self):
        self.contacts = {}
        self._load_registry()

    def _load_registry(self):
        """Loads registered face embeddings from disk or seeds default contacts."""
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    for name, data in raw_data.items():
                        self.contacts[name] = {
                            "name": data["name"],
                            "role": data.get("role", "Contact"),
                            "embedding": np.array(data["embedding"], dtype=np.float32),
                            "registered_at": data.get("registered_at", time.time())
                        }
                    print(f"[FaceSocialEngine] Loaded {len(self.contacts)} registered contacts.")
                    return
            except Exception as e:
                print(f"[FaceSocialEngine] Registry load error: {e}")

        # Seed initial project contacts
        self._seed_default_contacts()

    def _seed_default_contacts(self):
        """Seeds known project members as standard contacts."""
        defaults = [
            {"name": "Vikas Sir", "role": "Project Supervisor (Asst. Prof, SKIT)"},
            {"name": "Amit Tiwari", "role": "Project Partner"},
            {"name": "Deepak Choudhary", "role": "Project Partner"},
            {"name": "Deependra Singh", "role": "Project Partner"}
        ]
        for c in defaults:
            # Seed distinctive synthetic baseline embedding
            seed_vec = np.random.RandomState(abs(hash(c["name"])) % (2**31)).normal(0, 1.0, 128).astype(np.float32)
            seed_vec /= np.linalg.norm(seed_vec)
            self.contacts[c["name"]] = {
                "name": c["name"],
                "role": c["role"],
                "embedding": seed_vec,
                "registered_at": time.time()
            }

    def _save_registry(self):
        """Persists registered contacts to JSON."""
        try:
            serializable = {}
            for name, data in self.contacts.items():
                serializable[name] = {
                    "name": data["name"],
                    "role": data["role"],
                    "embedding": data["embedding"].tolist(),
                    "registered_at": data["registered_at"]
                }
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"[FaceSocialEngine] Registry save error: {e}")

    def extract_embedding(self, face_crop):
        """
        Extracts a 128-dimensional appearance & color-spatial descriptor vector.
        Normalized to unit length for fast cosine similarity comparison.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Resize to standardized 64x64 patch
        resized = cv2.resize(face_crop, (64, 64))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # 3D HSV Color Histogram (8x8x2 = 128 bins)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 2], [0, 180, 0, 256, 0, 256])
        vector = hist.flatten().astype(np.float32)

        # L2 Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def estimate_expression(self, face_crop):
        """Estimates basic facial emotion/expression (smiling, neutral, attentive)."""
        if face_crop is None or face_crop.shape[0] < 20:
            return "neutral"

        # Lower 35% of face represents mouth region
        h, w = face_crop.shape[:2]
        mouth_crop = face_crop[int(h * 0.65):, int(w * 0.2):int(w * 0.8)]
        if mouth_crop.size == 0:
            return "neutral"

        gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
        # Check variance and horizontal edges
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = np.var(sobel_h)

        if edge_energy > 400:
            return "smiling"
        elif edge_energy < 120:
            return "attentive"
        else:
            return "neutral"

    def identify_person(self, frame, person_bbox):
        """
        Extracts face from person bounding box [x1, y1, x2, y2],
        identifies against registered contacts, and estimates expression.
        Returns:
            dict containing:
                - recognized: bool
                - name: str
                - role: str
                - confidence: float
                - expression: str ('smiling', 'neutral', 'attentive')
                - face_bbox: [fx1, fy1, fx2, fy2]
                - greeting_en: str
                - greeting_hi: str
        """
        if frame is None or not person_bbox:
            return None

        x1, y1, x2, y2 = person_bbox
        h_box = y2 - y1
        w_box = x2 - x1

        # Face region corresponds to upper 30% of person bounding box
        fy1 = max(0, y1)
        fy2 = min(frame.shape[0], y1 + int(h_box * 0.32))
        fx1 = max(0, x1 + int(w_box * 0.15))
        fx2 = min(frame.shape[1], x2 - int(w_box * 0.15))

        if (fy2 - fy1) < 15 or (fx2 - fx1) < 15:
            return None

        face_crop = frame[fy1:fy2, fx1:fx2]
        emb = self.extract_embedding(face_crop)
        expression = self.estimate_expression(face_crop)

        if emb is None:
            return None

        best_match = None
        best_sim = 0.0

        for name, data in self.contacts.items():
            sim = float(np.dot(emb, data["embedding"]))
            if sim > best_sim:
                best_sim = sim
                best_match = data

        # Cosine similarity threshold for positive recognition
        MATCH_THRESHOLD = 0.72

        if best_match and best_sim >= MATCH_THRESHOLD:
            name = best_match["name"]
            role = best_match["role"]
            conf = round(best_sim, 2)
            expr_desc = f", {expression}" if expression == "smiling" else ""
            greeting_en = f"{name} ({role}) is in front of you{expr_desc}."
            greeting_hi = f"{name} आपके सामने हैं{expr_desc}।"

            return {
                "recognized": True,
                "name": name,
                "role": role,
                "confidence": conf,
                "expression": expression,
                "face_bbox": [fx1, fy1, fx2, fy2],
                "greeting_en": greeting_en,
                "greeting_hi": greeting_hi
            }

        return {
            "recognized": False,
            "name": "Unfamiliar Person",
            "role": "Visitor",
            "confidence": round(best_sim, 2),
            "expression": expression,
            "face_bbox": [fx1, fy1, fx2, fy2],
            "greeting_en": f"An unfamiliar person is ahead ({expression}).",
            "greeting_hi": f"कोई अपरिचित व्यक्ति आगे खड़ा है।"
        }

    def register_contact(self, name, role, face_crop):
        """Registers a new contact into the known face memory database."""
        emb = self.extract_embedding(face_crop)
        if emb is None:
            return False, "Could not extract face features from image"

        self.contacts[name] = {
            "name": name,
            "role": role,
            "embedding": emb,
            "registered_at": time.time()
        }
        self._save_registry()
        return True, f"Successfully registered {name} ({role})"

    def list_contacts(self):
        """Returns metadata list of all registered known contacts."""
        return [
            {
                "name": data["name"],
                "role": data["role"],
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["registered_at"]))
            }
            for data in self.contacts.values()
        ]
