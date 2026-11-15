"""
Caregiver "See What I See" Live Tele-Assistance Engine.
Enables real-time remote visual assistance and two-way intercom between
a visually impaired user and their caregiver/family.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import time

class TeleAssistEngine:
    def __init__(self):
        self.call_active = False
        self.session_id = None
        self.started_at = None
        self.last_frame = None
        self.last_frame_timestamp = None
        self.pending_caregiver_guidance = []
        self.signals = {"offer": None, "answer": None, "candidates": []}

    def start_call(self, session_id="live_teleassist"):
        """Initializes tele-assistance call."""
        self.call_active = True
        self.session_id = session_id
        self.started_at = time.time()
        self.signals = {"offer": None, "answer": None, "candidates": []}
        return {"status": "call_started", "session_id": session_id}

    def end_call(self):
        """Terminates tele-assistance call."""
        self.call_active = False
        self.last_frame = None
        self.pending_caregiver_guidance.clear()
        return {"status": "call_ended"}

    def update_frame(self, frame_base64):
        """Updates the latest camera view transmitted from user device."""
        self.last_frame = frame_base64
        self.last_frame_timestamp = time.time()
        return True

    def get_latest_frame(self):
        """Retrieves the latest view for the Caregiver Dashboard."""
        return {
            "call_active": self.call_active,
            "has_frame": self.last_frame is not None,
            "frame": self.last_frame,
            "timestamp": self.last_frame_timestamp
        }

    def post_caregiver_guidance(self, message):
        """Allows caregiver to speak directly into the user's earphones."""
        if not message or not message.strip():
            return False
        self.pending_caregiver_guidance.append({
            "message": message.strip(),
            "timestamp": time.time()
        })
        return True

    def fetch_pending_guidance(self):
        """User device polls for spoken directions sent by caregiver."""
        if not self.pending_caregiver_guidance:
            return []
        msgs = list(self.pending_caregiver_guidance)
        self.pending_caregiver_guidance.clear()
        return msgs

    def set_signal(self, signal_type, data):
        """WebRTC signaling exchange (offer, answer, candidate)."""
        if signal_type == "offer":
            self.signals["offer"] = data
        elif signal_type == "answer":
            self.signals["answer"] = data
        elif signal_type == "candidate":
            self.signals["candidates"].append(data)
        return True

    def get_signals(self):
        """Fetches pending signaling packets."""
        return self.signals
