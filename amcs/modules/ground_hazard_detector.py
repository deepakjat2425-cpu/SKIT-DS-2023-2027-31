"""
Ground Hazard Detector Module for visually impaired pedestrian safety.
Detects steps / stairs, curbs and potholes in the lower part of the picture (the ground just ahead).

A false "stairs" warning teaches the user to ignore real ones, so every check needs strong evidence:
  * steps and curbs are horizontal edges that run across most of the walkway, on a surface that is
    otherwise plain (grass, gravel, waves and crowds have edges everywhere and are skipped);
  * steps are three or more such edges, evenly spaced, each between two differently lit surfaces
    (tread and riser), so tile grout and painted stripes do not count; a curb is one such edge;
  * a pothole is a compact dark patch surrounded by plain, evenly lit ground.

Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np

from modules.nlp_describer import say_distance

EDGE_STRENGTH = 40          # Sobel response of a real step edge on the blurred image
ROW_COVERAGE = 0.5          # a step edge covers at least half of the visible floor width
CLUTTER_LIMIT = 0.07        # share of floor pixels with other (non-horizontal) edges; above = textured ground
STEP_SPACING_CV = 0.4       # step edges are evenly spaced (perspective shrinks far steps a little)
CURB_CONTRAST = 25          # grey levels between the ground on either side of a curb
STEP_CONTRAST = 12          # grey levels between tread and riser; a grout or painted line has none


class GroundHazardDetector:
    def __init__(self):
        # Distance calibration for lower visual plane
        # Lower y in frame = closer to user's feet
        self.min_hazard_area = 600
        self.max_hazard_distance_m = 3.5

    def estimate_ground_distance(self, y_pixel, frame_height):
        """
        Estimates ground distance (meters) based on perspective geometry.
        Bottom edge of frame (y=h) represents ~0.6m in front of feet.
        Horizon / upper ground cutoff (y=h*0.65) represents ~3.5m.
        """
        norm_y = max(0.0, min(1.0, (frame_height - y_pixel) / (frame_height * 0.35 + 1e-5)))
        dist = 0.6 + norm_y * (self.max_hazard_distance_m - 0.6)
        return round(float(dist), 1)

    def analyze_ground_plane(self, frame, object_boxes=None):
        """
        Analyzes the bottom 38% of the frame for steps, curbs and potholes.
        object_boxes: optional [x1, y1, x2, y2] boxes of detected objects (people, furniture...).
            Those areas are not floor, so they are excluded from the analysis; otherwise dark
            clothing or a table edge is mistaken for a pothole or a curb.
        Returns:
            annotated_frame: np.ndarray with detected ground hazards
            hazards: list of detected hazards with type, severity, distance, and advice
        """
        if frame is None:
            return None, []

        h, w = frame.shape[:2]
        ground_y_start = int(h * 0.62)
        ground_roi = frame[ground_y_start:h, 0:w]

        hazards = []
        annotated_frame = frame.copy()

        gray = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Areas covered by detected objects are not floor
        floor_mask = np.ones(blurred.shape, dtype=bool)
        for box in object_boxes or []:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            ry1, ry2 = max(0, y1 - ground_y_start), min(blurred.shape[0], y2 - ground_y_start)
            rx1, rx2 = max(0, x1), min(w, x2)
            if ry2 > ry1 and rx2 > rx1:
                floor_mask[ry1:ry2, rx1:rx2] = False
        if not floor_mask.any() or floor_mask.mean() < 0.25:
            return annotated_frame, []      # almost no visible floor: nothing reliable to analyse
        # Edges found inside (or right at the border of) an object box are ignored. The image itself is
        # left untouched: repainting those areas would create artificial edges at the box borders.
        self._valid = None
        if not floor_mask.all():
            self._valid = cv2.erode(floor_mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        valid = floor_mask if self._valid is None else self._valid > 0

        # Steps and curbs share one measurement: which rows hold a long horizontal edge
        bands, clutter = self._horizontal_edge_bands(blurred, valid)
        if clutter <= CLUTTER_LIMIT:
            if not self._detect_stairs(blurred, valid, bands, ground_y_start, w, hazards, annotated_frame):
                self._detect_curb(blurred, valid, bands, ground_y_start, w, hazards, annotated_frame)

        self._detect_potholes(blurred, ground_y_start, w, hazards, annotated_frame)

        hazards.sort(key=lambda x: x["distance_m"])
        return annotated_frame, hazards

    def _only_floor(self, binary):
        """Zeroes a binary/edge map wherever the floor is hidden by a detected object."""
        valid = getattr(self, "_valid", None)
        return binary if valid is None else cv2.bitwise_and(binary, valid)

    @staticmethod
    def _horizontal_edge_bands(gray_roi, valid):
        """
        Returns (bands, clutter).
        bands: [(center_row, coverage)] for rows where horizontal edges cross at least ROW_COVERAGE
               of the visible floor width (neighbouring rows of one edge are merged).
        clutter: share of floor pixels holding edges that are not part of such a horizontal edge.
        """
        gx = np.abs(cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3))
        strong = ((gx > EDGE_STRENGTH) | (gy > EDGE_STRENGTH)) & valid
        horizontal = (gy > EDGE_STRENGTH) & (gy > 2.0 * gx) & valid

        width = valid.sum(axis=1)
        usable = width >= 0.4 * gray_roi.shape[1]
        coverage = np.where(usable, horizontal.sum(axis=1) / np.maximum(width, 1), 0.0)
        # A blurred edge spreads over a few rows; take the best of each 3-row window
        padded = np.pad(coverage, 1)
        coverage = np.maximum.reduce([padded[:-2], padded[1:-1], padded[2:]])
        edge_rows = coverage >= ROW_COVERAGE

        bands, start = [], None
        for y, on in enumerate(list(edge_rows) + [False]):
            if on and start is None:
                start = y
            elif not on and start is not None:
                center = (start + y - 1) // 2
                if bands and center - bands[-1][0] <= 6:          # two sides of one thin line
                    bands[-1] = ((bands[-1][0] + center) // 2, max(bands[-1][1], float(coverage[start:y].max())))
                else:
                    bands.append((center, float(coverage[start:y].max())))
                start = None

        in_band = np.zeros(gray_roi.shape[0], dtype=bool)
        for center, _cov in bands:
            in_band[max(0, center - 4):center + 5] = True
        other = strong & ~in_band[:, None]
        clutter = float(other.sum()) / max(1, int(valid.sum()))
        return bands, clutter

    @staticmethod
    def _level_change(gray_roi, valid, center, reach_up, reach_down):
        """Difference in median brightness just above and just below an edge row."""
        roi_h = gray_roi.shape[0]
        a0, a1 = max(0, center - reach_up), max(0, center - 3)
        b0, b1 = min(roi_h, center + 3), min(roi_h, center + reach_down)
        above, below = gray_roi[a0:a1][valid[a0:a1]], gray_roi[b0:b1][valid[b0:b1]]
        if above.size == 0 or below.size == 0:
            return 0.0
        return abs(float(np.median(above)) - float(np.median(below)))

    def _detect_stairs(self, gray_roi, valid, bands, y_offset, frame_w, hazards, annotated):
        """Three or more evenly spaced edges across the walkway, each a change of surface: steps."""
        if len(bands) < 3:
            return False
        rows = [c for c, _cov in bands]
        spacing = np.diff(rows)
        roi_h = annotated.shape[0] - y_offset
        if spacing.min() < 6 or spacing.max() > 0.4 * roi_h:
            return False
        if spacing.std() / spacing.mean() > STEP_SPACING_CV:
            return False
        # Look only as far as halfway to the neighbouring edge on each side
        gaps = [int(spacing[0])] + [int(g) for g in spacing] + [int(spacing[-1])]
        changes = sum(self._level_change(gray_roi, valid, r, max(4, gaps[i] // 2), max(4, gaps[i + 1] // 2)) >= STEP_CONTRAST
                      for i, r in enumerate(rows))
        if changes < 2 * len(rows) / 3:
            return False

        nearest_y = max(rows) + y_offset
        dist_m = self.estimate_ground_distance(nearest_y, annotated.shape[0])
        severity = "hazard" if dist_m <= 1.5 else "caution"
        hazards.append({
            "type": "Steps",
            "category": "stairs",
            "distance_m": dist_m,
            "severity": severity,
            "description": f"Steps ahead, {say_distance(dist_m)}",
            "advice": f"Careful, steps ahead, {say_distance(dist_m)}."
        })
        cv2.rectangle(annotated, (0, nearest_y - 20), (frame_w, nearest_y + 10), (0, 140, 255), 2)
        cv2.putText(annotated, f"HAZARD: STEPS (~{dist_m}m)", (14, nearest_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        return True

    def _detect_curb(self, gray_roi, valid, bands, y_offset, frame_w, hazards, annotated):
        """One or two strong edges across the walkway where the ground level (brightness) changes."""
        if not 1 <= len(bands) <= 2:
            return
        roi_h = gray_roi.shape[0]
        for center, cov in sorted(bands, key=lambda b: -b[0]):           # nearest first
            if cov < 0.6:
                continue
            above = gray_roi[max(0, center - 18):max(0, center - 6)]
            below = gray_roi[min(roi_h, center + 6):min(roi_h, center + 18)]
            above_ok = valid[max(0, center - 18):max(0, center - 6)]
            below_ok = valid[min(roi_h, center + 6):min(roi_h, center + 18)]
            if not above_ok.any() or not below_ok.any():
                continue
            if abs(float(np.median(above[above_ok])) - float(np.median(below[below_ok]))) < CURB_CONTRAST:
                continue                                                  # a painted line, not a level change
            abs_y = center + y_offset
            dist_m = self.estimate_ground_distance(abs_y, annotated.shape[0])
            severity = "hazard" if dist_m <= 1.4 else "caution"
            hazards.append({
                "type": "Curb / Step",
                "category": "curb",
                "distance_m": dist_m,
                "severity": severity,
                "description": f"A curb or step ahead, {say_distance(dist_m)}",
                "advice": f"Careful, a curb or step ahead, {say_distance(dist_m)}."
            })
            color = (0, 0, 230) if severity == "hazard" else (0, 165, 255)
            cv2.line(annotated, (0, abs_y), (frame_w, abs_y), color, 3, cv2.LINE_AA)
            cv2.putText(annotated, f"CURB / STEP ~{dist_m}m", (8, max(20, abs_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
            return

    def _detect_potholes(self, gray_roi, y_offset, frame_w, hazards, annotated):
        """A compact dark patch on plain, evenly lit ground: a pothole or open drain."""
        roi_h = gray_roi.shape[0]
        # A pothole is a dark patch on an otherwise lit floor; on a dark floor this test is meaningless
        valid = getattr(self, "_valid", None)
        floor_level = float(np.median(gray_roi if valid is None else gray_roi[valid > 0])) if (valid is None or valid.any()) else 0.0
        if floor_level < 80:
            return
        _, thresh = cv2.threshold(gray_roi, 60, 255, cv2.THRESH_BINARY_INV)
        thresh = self._only_floor(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not self.min_hazard_area <= area <= (frame_w * roi_h * 0.25):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh) if bh > 0 else 0
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0
            # Must be fully enclosed (not touching the frame edge), compact and clearly darker
            enclosed = x > 2 and y > 2 and (x + bw) < frame_w - 2 and (y + bh) < roi_h - 2
            if not (0.5 <= aspect <= 3.5 and solidity >= 0.8 and enclosed):
                continue
            blob_mask = np.zeros(gray_roi.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            if float(cv2.mean(gray_roi, mask=blob_mask)[0]) >= floor_level * 0.5:
                continue
            # The ground around it must be plain and lit (not grass, water, a crowd or a shadow edge)
            ring = cv2.subtract(cv2.dilate(blob_mask, ring_kernel), blob_mask)
            ring = self._only_floor(ring)
            ring_px = gray_roi[ring > 0]
            if ring_px.size < 200:
                continue
            if float(np.percentile(ring_px, 10)) < floor_level * 0.6 or float(ring_px.std()) > 30:
                continue

            abs_y = y + y_offset
            dist_m = self.estimate_ground_distance(abs_y + bh, y_offset + roi_h)
            severity = "hazard" if dist_m <= 1.3 else "caution"
            hazards.append({
                "type": "Pothole",
                "category": "pothole",
                "distance_m": dist_m,
                "severity": severity,
                "description": f"A hole or dip in the ground ahead, {say_distance(dist_m)}",
                "advice": f"Careful, a hole in the ground ahead, {say_distance(dist_m)}."
            })
            color = (0, 0, 230) if severity == "hazard" else (0, 140, 255)
            cv2.rectangle(annotated, (x, abs_y), (x + bw, abs_y + bh), color, 2)
            cv2.putText(annotated, f"POTHOLE ~{dist_m}m", (x, max(20, abs_y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            break
