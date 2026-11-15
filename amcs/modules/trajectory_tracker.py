"""
Trajectory & Time-to-Collision (TTC) Dynamic Motion Tracker.
Adapts Advanced Driver Assistance Systems (ADAS) collision avoidance algorithms
for visually impaired assistive pedestrian navigation.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import time
import math
import numpy as np

class TrackedObject:
    def __init__(self, track_id, class_name, bbox, distance_m, spatial, timestamp):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.distance_m = distance_m
        self.spatial = spatial
        self.history = [(timestamp, bbox, distance_m, (bbox[0] + bbox[2]) / 2.0)]
        self.last_updated = timestamp
        self.hits = 1                   # frames this object has been seen in
        self.radial_velocity_mps = 0.0  # Positive = approaching user
        self.lateral_velocity_mps = 0.0 # Positive = moving right
        self.speed_kmh = 0.0
        self.ttc_seconds = None
        self.collision_risk = "SAFE"
        self.evasive_action = "none"

    def update(self, bbox, distance_m, spatial, timestamp):
        self.bbox = bbox
        self.spatial = spatial
        self.hits += 1
        cx = (bbox[0] + bbox[2]) / 2.0
        self.history.append((timestamp, bbox, distance_m, cx))
        if len(self.history) > 12:
            self.history.pop(0)

        # Approach speed from the distance trend over the last ~1.5 s, not frame-to-frame deltas:
        # monocular distance estimates jitter by tens of centimetres between frames, which would
        # otherwise produce false "collision" alarms for objects that are standing still.
        recent = [(t, d) for t, _b, d, _c in self.history if timestamp - t <= 1.5]
        velocity = 0.0
        if len(recent) >= 2:
            span = recent[-1][0] - recent[0][0]
            closing = recent[0][1] - recent[-1][1]          # positive if getting closer
            if span >= 0.5 and abs(closing) >= 0.6:
                if len(recent) >= 3:
                    ts = np.array([r[0] for r in recent]) - recent[0][0]
                    ds = np.array([r[1] for r in recent])
                    velocity = -float(np.polyfit(ts, ds, 1)[0])
                else:
                    velocity = closing / span
        self.radial_velocity_mps = velocity
        self.distance_m = distance_m

        # Estimate speed in km/h
        self.speed_kmh = max(0.0, round(abs(self.radial_velocity_mps) * 3.6, 1))

        # Time-to-Collision (TTC) Calculation: TTC = Distance / Approach_Velocity
        # Only meaningful if object is approaching (radial_velocity_mps > 0.35 m/s)
        if self.radial_velocity_mps > 0.35 and self.distance_m > 0.3:
            self.ttc_seconds = round(self.distance_m / self.radial_velocity_mps, 1)
        else:
            self.ttc_seconds = None

        # Collision Risk Assessment
        if self.ttc_seconds is not None:
            if self.ttc_seconds <= 2.8 and (self.spatial in ["ahead", "center"] or self.distance_m <= 2.0):
                self.collision_risk = "IMMINENT_HAZARD"
                # Determine best evasive steering direction
                if self.spatial == "left":
                    self.evasive_action = "step_right"
                elif self.spatial == "right":
                    self.evasive_action = "step_left"
                else:
                    self.evasive_action = "step_right"
            elif self.ttc_seconds <= 5.0:
                self.collision_risk = "APPROACHING_CAUTION"
                self.evasive_action = "prepare_to_swerve"
            else:
                self.collision_risk = "SAFE"
                self.evasive_action = "none"
        else:
            self.collision_risk = "SAFE"
            self.evasive_action = "none"

        self.last_updated = timestamp


class TrajectoryTracker:
    def __init__(self, max_missing_time=1.5):
        self.tracks = {}
        self.next_track_id = 1
        self.max_missing_time = max_missing_time
        self.view = None

    def set_view(self, view):
        """Forgets all tracks when the camera or frame size changes: old boxes belong to another view."""
        if view != self.view:
            self.tracks.clear()
            self.view = view

    def _centroid_dist(self, boxA, boxB):
        cxA = (boxA[0] + boxA[2]) / 2.0
        cyA = (boxA[1] + boxA[3]) / 2.0
        cxB = (boxB[0] + boxB[2]) / 2.0
        cyB = (boxB[1] + boxB[3]) / 2.0
        return math.hypot(cxA - cxB, cyA - cyB)

    def update_tracks(self, detections, timestamp=None):
        """
        Updates dynamic trajectory tracking for current detections list.
        Augments each detection dict with dynamic motion fields:
            - track_id: int
            - radial_velocity_mps: float
            - speed_kmh: float
            - ttc_seconds: float or None
            - collision_risk: 'IMMINENT_HAZARD' | 'APPROACHING_CAUTION' | 'SAFE'
            - evasive_action: str
        """
        if timestamp is None:
            timestamp = time.time()

        # Remove stale tracks that haven't been seen recently
        stale_ids = [tid for tid, trk in self.tracks.items() if (timestamp - trk.last_updated) > self.max_missing_time]
        for tid in stale_ids:
            del self.tracks[tid]

        matched_tracks = set()

        for det in detections:
            bbox = det["bbox"]
            cls_name = det["class"]
            dist_m = det["distance_m"]
            spatial = det["spatial"]

            # Find closest matching existing track of the same class
            best_tid = None
            best_dist = 180.0  # Max pixel centroid distance threshold

            for tid, trk in self.tracks.items():
                if tid in matched_tracks or trk.class_name != cls_name:
                    continue
                # Between two frames ~0.6 s apart a real object's box changes size only a little.
                # A much bigger/smaller box is a different object (or the camera turned to a new view),
                # and matching it would fake a sudden approach and a false collision warning.
                old_area = max(1.0, (trk.bbox[2] - trk.bbox[0]) * (trk.bbox[3] - trk.bbox[1]))
                new_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                if not (0.55 <= new_area / old_area <= 1.8):
                    continue
                # Same for the shape: a knee (wide box) does not turn into a standing person (tall box)
                old_aspect = max(1.0, trk.bbox[2] - trk.bbox[0]) / max(1.0, trk.bbox[3] - trk.bbox[1])
                new_aspect = max(1.0, bbox[2] - bbox[0]) / max(1.0, bbox[3] - bbox[1])
                if not (0.6 <= new_aspect / old_aspect <= 1.7):
                    continue
                d = self._centroid_dist(trk.bbox, bbox)
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid is not None:
                # Existing track found: update trajectory
                self.tracks[best_tid].update(bbox, dist_m, spatial, timestamp)
                matched_tracks.add(best_tid)
                trk = self.tracks[best_tid]
            else:
                # New track
                new_id = self.next_track_id
                self.next_track_id += 1
                trk = TrackedObject(new_id, cls_name, bbox, dist_m, spatial, timestamp)
                self.tracks[new_id] = trk
                matched_tracks.add(new_id)

            # A stationary object's distance estimate jitters by a few tens of centimetres from
            # frame to frame; report the median of the last sightings so it does not flip between
            # "caution" and "hazard". Something that is really approaching keeps its live value.
            if trk.hits >= 2 and trk.radial_velocity_mps <= 0.35:
                recent = [d for t, _b, d, _c in trk.history[-3:] if timestamp - t <= 1.5]
                if len(recent) >= 2:
                    det["distance_raw"] = dist_m
                    det["distance_m"] = round(float(np.median(recent)), 1)

            # Augment detection item with dynamic trajectory metrics
            det["track_id"] = trk.track_id
            det["speed_kmh"] = trk.speed_kmh
            det["radial_velocity_mps"] = round(trk.radial_velocity_mps, 2)
            det["ttc_seconds"] = trk.ttc_seconds
            det["collision_risk"] = trk.collision_risk
            det["evasive_action"] = trk.evasive_action
            det["hits"] = trk.hits

        # Identify most urgent dynamic threat across all detections
        urgent_dynamic = [d for d in detections if d.get("collision_risk") == "IMMINENT_HAZARD"]
        if urgent_dynamic:
            urgent_dynamic.sort(key=lambda x: x["ttc_seconds"] if x["ttc_seconds"] is not None else 99)
            primary_hazard = urgent_dynamic[0]
        else:
            primary_hazard = None

        return detections, primary_hazard
