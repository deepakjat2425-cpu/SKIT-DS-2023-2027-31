"""
Indoor Door, Handle & Room Sign Locator Engine.
Enables indoor building navigation for visually impaired individuals where GPS fails.
Detects doors, open doorways, handle/knob position (left/right), and room signs.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np
import math

class IndoorDoorLocator:
    def __init__(self):
        self.min_door_aspect = 1.35  # Height / Width ratio for doors
        self.max_door_aspect = 3.2

    def analyze_frame_for_doors(self, frame, ocr_items=None, object_boxes=None):
        """
        Scans an indoor image frame for doors, doorways, handle positions, and door signs.
        Returns:
            annotated_frame: np.ndarray with door reticle and handle indicator
            doors: list of detected door objects
        """
        if frame is None:
            return None, []

        h, w = frame.shape[:2]
        doors = []
        annotated = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 130)

        # Connect vertical frame contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_door_area = (w * h) * 0.08  # Door must occupy at least 8% of frame
        max_door_area = (w * h) * 0.85

        for c in contours:
            area = cv2.contourArea(c)
            if min_door_area <= area <= max_door_area:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 0

                # A tall contour that is mostly a detected object (person, fridge, wardrobe...) is not a door
                if object_boxes and any(
                    max(0, min(x + bw, b[2]) - max(x, b[0])) * max(0, min(y + bh, b[3]) - max(y, b[1])) > 0.5 * bw * bh
                    for b in object_boxes
                ):
                    continue

                if self.min_door_aspect <= aspect <= self.max_door_aspect:
                    # Door candidate found
                    door_roi = gray[y:y+bh, x:x+bw]
                    door_state, openness_ratio = self._classify_door_state(door_roi)
                    handle_side, handle_coords = self._locate_handle(door_roi, x, y, bw, bh)

                    # Estimate distance in meters (based on typical 2.0m standard door height)
                    dist_m = round(max(0.7, min(6.0, (2.0 * 580.0) / max(10, bh))), 1)

                    # Correlate nearby room sign from OCR items
                    sign_text = self._find_nearby_door_sign(ocr_items, x, y, bw, bh)

                    door_obj = {
                        "bbox": [x, y, x + bw, y + bh],
                        "state": door_state,
                        "distance_m": dist_m,
                        "handle_side": handle_side,
                        "handle_coords": handle_coords,
                        "sign_text": sign_text,
                        "guidance": self._format_door_guidance(door_state, dist_m, handle_side, sign_text)
                    }
                    doors.append(door_obj)

                    # Draw stylish AR HUD door reticle
                    self._draw_door_overlay(annotated, door_obj)
                    break

        return annotated, doors

    def _classify_door_state(self, door_roi):
        """Distinguishes between closed door, partially open door, and open doorway."""
        # Mean intensity and standard deviation inside door ROI
        mean_val = np.mean(door_roi)
        # Check central vertical opening slit
        h, w = door_roi.shape
        center_strip = door_roi[:, int(w * 0.35):int(w * 0.65)]
        center_mean = np.mean(center_strip)

        if center_mean < 45 and mean_val < 65:
            return "open", 1.0
        elif center_mean < mean_val * 0.60:
            return "partially_open", 0.5
        else:
            return "closed", 0.0

    def _locate_handle(self, door_roi, abs_x, abs_y, bw, bh):
        """Locates door knob or lever handle along the middle-vertical third."""
        roi_h, roi_w = door_roi.shape
        handle_y_start = int(roi_h * 0.40)
        handle_y_end = int(roi_h * 0.65)
        mid_zone = door_roi[handle_y_start:handle_y_end, :]

        # Look for local high-contrast contour in left vs right outer margins
        margin_w = int(roi_w * 0.28)
        left_margin = mid_zone[:, :margin_w]
        right_margin = mid_zone[:, roi_w - margin_w:]

        left_edges = np.sum(cv2.Canny(left_margin, 50, 150))
        right_edges = np.sum(cv2.Canny(right_margin, 50, 150))

        if right_edges >= left_edges:
            handle_side = "right"
            handle_x = abs_x + bw - int(margin_w * 0.5)
        else:
            handle_side = "left"
            handle_x = abs_x + int(margin_w * 0.5)

        handle_y = abs_y + int((handle_y_start + handle_y_end) / 2.0)
        return handle_side, [handle_x, handle_y]

    def _find_nearby_door_sign(self, ocr_items, dx, dy, dw, dh):
        """Matches extracted text tokens above or near the door header."""
        if not ocr_items:
            return None

        for item in ocr_items:
            pts = item.get("bbox", [])
            if pts and len(pts) >= 4:
                tx = pts[0][0]
                ty = pts[0][1]
                # If text is situated near door top header (+/- 120 pixels)
                if (dx - 40) <= tx <= (dx + dw + 40) and (dy - 120) <= ty <= (dy + int(dh * 0.35)):
                    return item.get("text", "").strip()
        return None

    def _format_door_guidance(self, state, dist_m, handle_side, sign_text):
        """Formats natural spoken navigational guidance for the door."""
        sign_str = f" Sign reads: {sign_text}." if sign_text else ""
        if state == "open":
            return f"Open doorway {dist_m} meters directly ahead. Path through door is clear.{sign_str}"
        elif state == "partially_open":
            return f"Doorway is ajar {dist_m} meters ahead. Handle on the {handle_side}.{sign_str}"
        else:
            return f"Closed door {dist_m} meters ahead. Handle is on the {handle_side} side.{sign_str}"

    def _draw_door_overlay(self, frame, door_obj):
        """Draws chamfered door bounding bracket and handle reticle on frame."""
        x1, y1, x2, y2 = door_obj["bbox"]
        state = door_obj["state"]
        color = (52, 211, 153) if state == "open" else (99, 102, 241)  # Emerald or Indigo

        # Draw door boundary
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw handle target
        hx, hy = door_obj["handle_coords"]
        cv2.circle(frame, (hx, hy), 8, (0, 240, 255), -1)
        cv2.circle(frame, (hx, hy), 14, (0, 240, 255), 2)

        # Header tag
        label = f"DOOR: {state.upper()} (~{door_obj['distance_m']}m) | Handle {door_obj['handle_side'].upper()}"
        if door_obj["sign_text"]:
            label += f" | {door_obj['sign_text']}"
        cv2.putText(frame, label, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
