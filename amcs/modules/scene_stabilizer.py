"""
Temporal confirmation for single-frame heuristics.
Edge/contour based detectors (potholes, curbs, stairs, doors) fire on random frames because of
shadows, clothing or furniture edges. A finding is only reported once it has been seen in most of
the recent frames, which removes flicker and one-off false alarms.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import time
from collections import Counter, deque


class TemporalConfirmer:
    def __init__(self, window=4, need=3, max_gap_s=2.0):
        self.need = need
        self.max_gap_s = max_gap_s
        self.history = deque(maxlen=window)
        self.last_t = 0.0

    def update(self, keys, timestamp=None):
        """Records the keys seen in this frame; returns the set of keys that are confirmed."""
        t = time.time() if timestamp is None else timestamp
        if t - self.last_t > self.max_gap_s:
            self.history.clear()          # camera paused or user moved on: start fresh
        self.last_t = t
        self.history.append(set(keys))
        counts = Counter(k for frame_keys in self.history for k in frame_keys)
        return {k for k, n in counts.items() if n >= self.need}

    def reset(self):
        self.history.clear()
