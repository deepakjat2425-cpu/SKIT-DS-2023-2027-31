"""
Pedestrian Navigation & Turn-by-Turn Wayfinding Engine.
Voice-guided walking directions for visually impaired users.

  * Destinations: places the user saved ("save this place as home"), campus landmarks, or any
    address / landmark found on OpenStreetMap near the user.
  * Routes: real walking routes (OpenStreetMap foot routing) with turn-by-turn steps; without
    internet the user is guided straight towards the destination instead.
  * Turns are told relative to the way the user is facing (phone compass): "Turn to your left",
    "In 30 meters, turn right", "Your destination is at 2 o'clock, 80 meters away".
  * Leaving the route is noticed and a new route is planned from where the user is.

Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import json
import math
import os
import re
import threading
import time

import requests

# Known SKIT Jaipur Campus Geolocation Waypoints & Nodes
SKIT_WAYPOINTS = {
    "main_gate": {
        "name": "Main Entrance Gate",
        "aliases": ["gate", "main gate", "entrance", "exit"],
        "lat": 26.82285,
        "lon": 75.86475,
        "description": "SKIT Main Entrance Gate, Ramnagaria road"
    },
    "admin_block": {
        "name": "Administrative Block",
        "aliases": ["admin", "admin block", "office", "principal office"],
        "lat": 26.82340,
        "lon": 75.86530,
        "description": "Administrative Block & Reception"
    },
    "central_library": {
        "name": "Central Library",
        "aliases": ["library", "central library", "books", "reading room"],
        "lat": 26.82390,
        "lon": 75.86560,
        "description": "Central Library & Digital Knowledge Resource Centre"
    },
    "cse_block": {
        "name": "Computer Science & Engineering Dept",
        "aliases": ["cse", "computer science", "cse block", "it dept", "cse lab"],
        "lat": 26.82420,
        "lon": 75.86510,
        "description": "Department of Computer Science & Engineering"
    },
    "civil_block": {
        "name": "Civil & Mechanical Engineering Block",
        "aliases": ["civil", "civil block", "mech", "mechanical"],
        "lat": 26.82450,
        "lon": 75.86460,
        "description": "Civil & Mechanical Engineering Block"
    },
    "cafeteria": {
        "name": "Campus Cafeteria & Food Court",
        "aliases": ["canteen", "cafeteria", "cafe", "food", "mess"],
        "lat": 26.82310,
        "lon": 75.86610,
        "description": "Campus Cafeteria & Student Food Court"
    },
    "boys_hostel": {
        "name": "Boys Hostel Campus",
        "aliases": ["boys hostel", "hostel", "bh"],
        "lat": 26.82510,
        "lon": 75.86580,
        "description": "Boys Residence Complex"
    },
    "girls_hostel": {
        "name": "Girls Hostel Campus",
        "aliases": ["girls hostel", "gh"],
        "lat": 26.82240,
        "lon": 75.86650,
        "description": "Girls Residence Complex"
    },
    "pharmacy": {
        "name": "Health Center & Pharmacy",
        "aliases": ["pharmacy", "medical", "doctor", "dispensary", "health"],
        "lat": 26.82360,
        "lon": 75.86490,
        "description": "Campus Dispensary & Medical Center"
    },
    "sports_ground": {
        "name": "Sports Ground & Gymnasium",
        "aliases": ["ground", "sports", "gym", "cricket", "basketball"],
        "lat": 26.82480,
        "lon": 75.86380,
        "description": "Outdoor Sports Ground & Athletic Track"
    }
}

EARTH_RADIUS_M = 6371000.0
USER_AGENT = "AMCS-SKIT-assistive-navigation/3.2"
ROUTER_URL = "https://routing.openstreetmap.de/routed-foot/route/v1/foot/{lon1},{lat1};{lon2},{lat2}"
GEOCODER_URL = "https://nominatim.openstreetmap.org/search"
REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
COMPASS_WORDS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
ONLINE_TIMEOUT_S = 6
MAX_WALK_KM = 30            # farther than this is not a walking destination
ARRIVE_RADIUS_M = 10        # within this of the destination (or the GPS accuracy, up to 20 m) = arrived
OFF_ROUTE_M = 30            # farther than this from the route (twice in a row) = a new route is planned
TURN_NOW_M = 12             # "Now, turn left"
TURN_SOON_M = 40            # "In 30 meters, turn left"
FACE_TOLERANCE_DEG = 50     # facing further off than this from the way to walk = "Turn to your left"


# ------------------------------------------------------------------ geometry
def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass direction from point 1 to point 2, 0-360 (0 = north, 90 = east)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def relative_angle(target_bearing, heading):
    """How far to turn from `heading` to face `target_bearing`: -180..180, positive = to the right."""
    return (target_bearing - heading + 540) % 360 - 180


def clock_position(rel):
    """Relative angle -> clock face for the user ('12' = straight ahead, '3' = right)."""
    hour = round(rel / 30) % 12
    return 12 if hour == 0 else hour


def direction_phrase(rel):
    """'straight ahead', 'slightly to your left', 'to your right', 'behind you'..."""
    side = "right" if rel > 0 else "left"
    a = abs(rel)
    if a <= 20:
        return "straight ahead"
    if a <= 60:
        return f"slightly to your {side}"
    if a <= 120:
        return f"to your {side}"
    if a <= 160:
        return f"behind you, to the {side}"
    return "behind you"


def round_distance(m):
    """Spoken distances: 7 -> 5, 23 -> 20, 140 -> 140, 1260 -> 1.3 kilometers."""
    if m >= 1000:
        return f"{m / 1000:.1f} kilometers"
    step = 5 if m < 50 else 10
    return f"{max(step, int(round(m / step) * step))} meters"


def turn_modifier(bend):
    """Change of direction in degrees (positive = right) -> OSRM-style modifier."""
    side = "right" if bend > 0 else "left"
    a = abs(bend)
    if a <= 30:
        return "straight"
    if a <= 60:
        return f"slight {side}"
    if a <= 140:
        return side
    return f"sharp {side}" if a <= 170 else "uturn"


class _LocalFrame:
    """Flat x/y meters around a reference point; plenty accurate over a walking route."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * EARTH_RADIUS_M

    def xy(self, lat, lon):
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky


# ------------------------------------------------------------------ online services
def osrm_foot_route(origin, dest):
    """
    Walking route from OpenStreetMap (FOSSGIS foot router). origin / dest: (lat, lon).
    Returns {"coords": [(lat, lon), ...], "steps": [...], "distance_m"} or raises.
    """
    url = ROUTER_URL.format(lat1=origin[0], lon1=origin[1], lat2=dest[0], lon2=dest[1])
    res = requests.get(url, params={"overview": "full", "geometries": "geojson", "steps": "true"},
                       headers={"User-Agent": USER_AGENT}, timeout=ONLINE_TIMEOUT_S)
    res.raise_for_status()
    data = res.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"no route: {data.get('code')}")
    route = data["routes"][0]
    steps = []
    for leg in route["legs"]:
        for s in leg["steps"]:
            m = s["maneuver"]
            steps.append({
                "type": m.get("type", "turn"),
                "modifier": m.get("modifier", "straight"),
                "name": s.get("name") or "",
                "exit": m.get("exit"),
                "location": (m["location"][1], m["location"][0]),
                "distance_m": float(s.get("distance", 0.0)),
            })
    coords = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    return {"coords": coords, "steps": steps, "distance_m": float(route["distance"])}


def nominatim_search(query, near):
    """A place matching `query`, preferring results near `near` (lat, lon). Returns dict or None."""
    lat, lon = near
    params = {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 0,
              "viewbox": f"{lon - 0.3},{lat + 0.3},{lon + 0.3},{lat - 0.3}", "bounded": 1}
    res = requests.get(GEOCODER_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=ONLINE_TIMEOUT_S)
    res.raise_for_status()
    found = res.json()
    if not found:
        return None
    hit = found[0]
    full = hit.get("display_name", query)
    return {"name": hit.get("name") or full.split(",")[0], "lat": float(hit["lat"]), "lon": float(hit["lon"]),
            "description": full}


def nominatim_reverse(lat, lon):
    """Street and area at a position: {"road", "area", "city"} (any may be missing) or None."""
    params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 17, "addressdetails": 1}
    res = requests.get(REVERSE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=ONLINE_TIMEOUT_S)
    res.raise_for_status()
    addr = res.json().get("address") or {}
    if not addr:
        return None
    return {
        "road": addr.get("road") or addr.get("pedestrian") or addr.get("footway") or addr.get("path"),
        "area": addr.get("neighbourhood") or addr.get("suburb") or addr.get("quarter") or addr.get("amenity"),
        "city": addr.get("city") or addr.get("town") or addr.get("village"),
    }


# ------------------------------------------------------------------ saved places
class SavedPlaces:
    """Places the user saved by name ('home', 'office'...), kept in a small JSON file."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._places = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for p in json.load(f).get("places", []):
                        self._places[self.key(p["name"])] = p
            except (OSError, ValueError, KeyError) as e:
                print(f"[Navigation] Saved places not loaded: {e}")

    @staticmethod
    def key(name):
        return re.sub(r"\s+", " ", str(name)).strip().lower()

    def all(self):
        return sorted(self._places.values(), key=lambda p: p["name"].lower())

    def save(self, name, lat, lon):
        name = re.sub(r"\s+", " ", str(name)).strip()[:40]
        if not name:
            raise ValueError("A place needs a name.")
        place = {"name": name, "lat": round(float(lat), 6), "lon": round(float(lon), 6), "saved_at": int(time.time())}
        with self._lock:
            self._places[self.key(name)] = place
            self._write()
        return place

    def remove(self, name):
        with self._lock:
            removed = self._places.pop(self.key(name), None) is not None
            if removed:
                self._write()
        return removed

    def find(self, query):
        """Exact name, or a saved name said inside the query ('my home' -> 'home')."""
        q = self.key(query)
        if q in self._places:
            return self._places[q]
        words = f" {q} "
        hits = [p for k, p in self._places.items() if f" {k} " in words]
        return max(hits, key=lambda p: len(p["name"])) if hits else None

    def _write(self):
        if not self.path:
            return
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"places": list(self._places.values())}, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ spoken maneuvers
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
TURN_WORDS = {
    "uturn": "turn around", "sharp right": "turn sharply right", "right": "turn right",
    "slight right": "bear slightly right", "straight": "continue straight", "slight left": "bear slightly left",
    "left": "turn left", "sharp left": "turn sharply left",
}


def maneuver_phrase(step):
    """'turn left onto MI Road', 'at the roundabout, take the second exit', 'keep right at the fork'."""
    kind, mod = step.get("type", "turn"), step.get("modifier", "straight")
    name = step.get("name", "")
    if kind in ("roundabout", "rotary", "roundabout turn", "exit roundabout"):
        exit_no = step.get("exit")
        text = f"at the roundabout, take the {ORDINALS.get(exit_no, 'next')} exit"
    elif kind == "fork":
        text = f"keep {'left' if 'left' in mod else 'right'} at the fork"
    elif kind == "end of road":
        text = f"at the end of the road, {TURN_WORDS.get(mod, 'turn')}"
    else:
        text = TURN_WORDS.get(mod, "continue straight")
    if name and kind not in ("arrive",):
        text += f" onto {name}"
    return text


class NavigationAssistant:
    def __init__(self, places_file=None, online=True, router=None, geocoder=None, reverse_geocoder=None):
        self.places = SavedPlaces(places_file)
        self.online = online
        self._router = router or osrm_foot_route
        self._geocoder = geocoder or nominatim_search
        self._reverse = reverse_geocoder or nominatim_reverse
        self.active_route = None
        self.current_step_idx = 0
        self.destination_name = None

    # Kept for callers of the earlier API
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance between two GPS coordinates in meters."""
        return round(haversine_m(lat1, lon1, lat2, lon2), 1)

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculates compass heading from origin to destination in degrees (0-360)."""
        return round(bearing_deg(lat1, lon1, lat2, lon2), 1)

    # ------------------------------------------------------------------ destinations
    def resolve_destination(self, query, near=None):
        """
        Saved place, then campus landmark, then (when `near` is given and online) a search of
        OpenStreetMap around the user. Returns (key, place) or (None, None).
        """
        q = re.sub(r"\s+", " ", (query or "")).strip().lower()
        if not q:
            return None, None
        saved = self.places.find(q)
        if saved:
            return f"saved:{SavedPlaces.key(saved['name'])}", dict(saved, description="Saved place", source="saved")

        if q in SKIT_WAYPOINTS:
            return q, SKIT_WAYPOINTS[q]
        # The longest alias said wins: "girls hostel" must not stop at "hostel" (boys hostel)
        best = None
        for key, wp in SKIT_WAYPOINTS.items():
            for alias in [wp["name"].lower()] + wp.get("aliases", []):
                if re.search(rf"\b{re.escape(alias)}\b", q) or (len(q) >= 4 and re.search(rf"\b{re.escape(q)}\b", alias)):
                    if best is None or len(alias) > best[0]:
                        best = (len(alias), key, wp)
        if best:
            return best[1], best[2]

        if near is not None and self.online:
            try:
                found = self._geocoder(query, near)
            except Exception as e:                              # offline or service busy
                print(f"[Navigation] Place search failed: {e}")
                found = None
            if found:
                return "search", dict(found, source="search")
        return None, None

    # ------------------------------------------------------------------ planning
    def plan_pedestrian_route(self, origin_lat, origin_lon, dest_query, heading=None):
        """
        Plans a walking route from the user's position to the destination.
        heading: the way the user faces (compass degrees), when the phone knows it.
        """
        dest_key, dest = self.resolve_destination(dest_query, near=(origin_lat, origin_lon))
        if not dest:
            saved = ", ".join(p["name"] for p in self.places.all()[:4])
            hint = f" Your saved places: {saved}." if saved else ""
            return {"status": "error",
                    "message": f"I could not find '{dest_query}'.{hint} You can also say a landmark or an address."}

        straight = haversine_m(origin_lat, origin_lon, dest["lat"], dest["lon"])
        if straight > MAX_WALK_KM * 1000:
            return {"status": "error",
                    "message": f"{dest['name']} is {straight / 1000:.0f} kilometers away, too far for walking directions."}

        route = None
        if self.online and straight > ARRIVE_RADIUS_M:
            try:
                route = self._router((origin_lat, origin_lon), (dest["lat"], dest["lon"]))
            except Exception as e:                              # offline or no walkable route
                print(f"[Navigation] Walking route unavailable ({e}); guiding straight to the destination.")
        self._start(dest, origin_lat, origin_lon, route)

        r = self.active_route
        total = r["total_m"]
        minutes = max(1, round(total / 75))                     # about 4.5 km/h
        first = self._first_instruction(origin_lat, origin_lon, heading)
        summary = f"Route to {dest['name']}: {round_distance(total)}, about {minutes} minute{'s' if minutes > 1 else ''} walk."
        if r["mode"] == "direct":
            summary += " I could not get a street route, so I will guide you straight towards it."
        return {
            "status": "success",
            "destination": dest["name"],
            "destination_source": dest.get("source", "campus"),
            "total_distance_m": round(total, 1),
            "bearing_deg": round(bearing_deg(origin_lat, origin_lon, dest["lat"], dest["lon"]), 1),
            "mode": r["mode"],
            "steps": [{"step_index": i + 1, "instruction": s["instruction"], "distance_m": round(s["leg_m"]),
                       "maneuver": s["maneuver"]} for i, s in enumerate(r["steps"])],
            "initial_instruction": f"{summary} {first}",
            "path": [[round(a, 6), round(b, 6)] for a, b in r["coords"]],
        }

    def _start(self, dest, origin_lat, origin_lon, route):
        if route and len(route.get("coords", [])) >= 2:
            coords, mode = list(route["coords"]), "route"
            # A street changing its name while the user walks straight on is not worth announcing
            raw_steps = [s for s in route.get("steps", [])
                         if not (s.get("type") in ("new name", "continue", "notification")
                                 and s.get("modifier", "straight") == "straight")]
        else:
            coords, mode = [(origin_lat, origin_lon), (dest["lat"], dest["lon"])], "direct"
            raw_steps = [{"type": "depart", "modifier": "straight", "location": coords[0]},
                         {"type": "arrive", "modifier": "straight", "location": coords[1]}]
        if mode == "route":
            # The router starts and ends on the nearest street; the user may be off it (a campus,
            # a courtyard), so the walk to and from the street is part of the route
            if haversine_m(origin_lat, origin_lon, *coords[0]) > 3:
                street_start = coords[0]
                coords.insert(0, (origin_lat, origin_lon))
                # Where the user's path meets the street, the way on may bend: say so as a turn
                nxt = next((pt for pt in coords[2:] if haversine_m(*street_start, *pt) > 5), None)
                bend = 0.0 if nxt is None else relative_angle(
                    bearing_deg(*street_start, *nxt), bearing_deg(origin_lat, origin_lon, *street_start))
                if abs(bend) > 30:      # the street's start becomes a turn
                    raw_steps = [dict(st, type="turn", modifier=turn_modifier(bend)) if st.get("type") == "depart" else st
                                 for st in raw_steps]
                else:                   # straight on: one walk from the user's position
                    raw_steps = [st for st in raw_steps if st.get("type") != "depart"]
                raw_steps = [{"type": "depart", "modifier": "straight", "location": coords[0]}] + raw_steps
            if haversine_m(dest["lat"], dest["lon"], *coords[-1]) > 3:
                coords.append((dest["lat"], dest["lon"]))
                for st in raw_steps:
                    if st.get("type") == "arrive":
                        st["location"] = coords[-1]
        frame = _LocalFrame(coords[0][0], coords[0][1])
        pts = [frame.xy(a, b) for a, b in coords]
        cum = [0.0]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            cum.append(cum[-1] + math.hypot(x2 - x1, y2 - y1))
        steps = []
        for s in raw_steps:
            along, _off = self._snap(frame, pts, cum, s["location"], 0.0)
            steps.append(dict(s, along_m=along))
        steps.sort(key=lambda s: s["along_m"])
        for i, s in enumerate(steps):
            nxt = steps[i + 1]["along_m"] if i + 1 < len(steps) else cum[-1]
            s["leg_m"] = max(0.0, nxt - s["along_m"])
            if s["type"] == "depart":
                s["maneuver"], s["instruction"] = "depart", f"Walk {round_distance(s['leg_m'])}."
            elif s["type"] == "arrive":
                side = s.get("modifier", "")
                where = f" It is on your {side.split()[-1]}." if side.split() and side.split()[-1] in ("left", "right") else ""
                s["maneuver"], s["instruction"] = "arrive", f"You have reached {dest['name']}.{where}"
            else:
                phrase = maneuver_phrase(s)
                s["maneuver"] = s.get("modifier", "straight")
                s["instruction"] = f"{phrase[0].upper()}{phrase[1:]}, then walk {round_distance(s['leg_m'])}."
        self.active_route = {
            "destination_name": dest["name"], "destination_coords": [dest["lat"], dest["lon"]],
            "mode": mode, "coords": coords, "frame": frame, "pts": pts, "cum": cum, "total_m": cum[-1],
            "steps": steps, "along_m": 0.0, "off_count": 0, "reroutes": 0, "start_time": time.time(),
        }
        self.current_step_idx = 0
        self.destination_name = dest["name"]

    @staticmethod
    def _snap(frame, pts, cum, latlon, min_along):
        """Distance along the route of the point nearest to `latlon` (not behind `min_along`), and how far off it is."""
        px, py = frame.xy(*latlon)
        best = (float("inf"), 0.0)
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts, pts[1:])):
            if cum[i + 1] < min_along:
                continue
            dx, dy = x2 - x1, y2 - y1
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg2))
            off = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if off < best[0]:
                best = (off, cum[i] + t * math.sqrt(seg2))
        return best[1], best[0]

    def _point_at(self, along):
        """(lat, lon) of the point `along` meters down the route."""
        r = self.active_route
        cum, coords = r["cum"], r["coords"]
        along = max(0.0, min(r["total_m"], along))
        for i in range(len(cum) - 1):
            if cum[i + 1] >= along:
                seg = cum[i + 1] - cum[i]
                t = 0.0 if seg == 0 else (along - cum[i]) / seg
                (a1, b1), (a2, b2) = coords[i], coords[i + 1]
                return a1 + t * (a2 - a1), b1 + t * (b2 - b1)
        return coords[-1]

    def _first_instruction(self, lat, lon, heading):
        r = self.active_route
        first = r["steps"][0]
        target = self._point_at(min(r["total_m"], 15.0))
        walk = round_distance(first["leg_m"] if r["mode"] == "route" else r["total_m"])
        then = ""
        if r["mode"] == "route" and len(r["steps"]) > 2:
            then = f", then {maneuver_phrase(r['steps'][1])}"
        if heading is None:
            return f"Walk {walk}{then}. Hold the phone upright, facing forward, so I can tell you which way to turn."
        rel = relative_angle(bearing_deg(lat, lon, *target), heading)
        if abs(rel) <= 20:
            return f"Walk straight ahead for {walk}{then}."
        return f"Turn {direction_phrase(rel).replace('behind you', 'around')}, then walk {walk}{then}."

    # ------------------------------------------------------------------ live guidance
    def update_navigation_progress(self, user_lat, user_lon, heading=None, accuracy=None):
        """
        Guidance for the user's current position (and facing direction, when known).
        Returns status 'navigating' / 'arrived' with the instruction to speak and an
        announce_key that changes only when the instruction means something new.
        """
        r = self.active_route
        if not r:
            return None
        dest_lat, dest_lon = r["destination_coords"]
        dist_dest = haversine_m(user_lat, user_lon, dest_lat, dest_lon)
        arrive_radius = max(ARRIVE_RADIUS_M, min(20.0, accuracy or 0.0))
        if dist_dest <= arrive_radius:
            arrive = r["steps"][-1]["instruction"] if r["steps"][-1]["type"] == "arrive" else f"You have arrived at {r['destination_name']}."
            self.active_route = None
            return {"status": "arrived", "instruction": arrive.replace("You have reached", "You have arrived at"),
                    "dist_remaining_m": round(dist_dest, 1), "is_completed": True, "announce_key": "arrived"}

        along, off = self._snap(r["frame"], r["pts"], r["cum"], (user_lat, user_lon), r["along_m"] - 30.0)
        r["along_m"] = max(r["along_m"], along) if off < OFF_ROUTE_M else r["along_m"]

        # Left the route: plan a new one from here (street routes only)
        if r["mode"] == "route" and off > max(OFF_ROUTE_M, 1.5 * (accuracy or 0.0)):
            r["off_count"] += 1
            if r["off_count"] >= 2 and r["reroutes"] < 5:
                dest = {"name": r["destination_name"], "lat": dest_lat, "lon": dest_lon}
                reroutes = r["reroutes"] + 1
                try:
                    route = self._router((user_lat, user_lon), (dest_lat, dest_lon)) if self.online else None
                except Exception:
                    route = None
                if route:
                    self._start(dest, user_lat, user_lon, route)
                    self.active_route["reroutes"] = reroutes
                    text = f"You have left the route. New route: {self._first_instruction(user_lat, user_lon, heading)}"
                    return self._guidance(text, f"reroute:{reroutes}", dist_dest, None, 0, urgent=True)
                r["off_count"] = 0
        else:
            r["off_count"] = 0

        if r["mode"] == "direct":
            rel = relative_angle(bearing_deg(user_lat, user_lon, dest_lat, dest_lon), heading) if heading is not None else None
            remaining = round_distance(dist_dest)
            if rel is None:
                return self._guidance(f"Keep going towards {r['destination_name']}, {remaining} away.",
                                      f"direct:{int(dist_dest // 25)}", dist_dest, None, 0)
            clock = clock_position(rel)
            return self._guidance(f"{r['destination_name']} is at {clock} o'clock, {remaining} away.",
                                  f"direct:{clock}:{int(dist_dest // 25)}", dist_dest, rel, 0)

        # Next turn on the street route
        upcoming = [(i, s) for i, s in enumerate(r["steps"]) if s["type"] != "depart" and s["along_m"] > r["along_m"] + 3]
        idx, step = upcoming[0] if upcoming else (len(r["steps"]) - 1, r["steps"][-1])
        self.current_step_idx = idx
        to_turn = max(0.0, step["along_m"] - r["along_m"])
        remaining_route = max(0.0, r["total_m"] - r["along_m"])

        # Which way to walk now: a point a little further along the route, but not round the next
        # corner (that is what the turn instruction is for)
        rel = None
        turning_now = step["type"] != "arrive" and to_turn <= TURN_NOW_M
        if heading is not None and not turning_now:
            target = self._point_at(r["along_m"] + min(12.0, max(to_turn, 5.0)))
            rel = relative_angle(bearing_deg(user_lat, user_lon, *target), heading)
            if abs(rel) > FACE_TOLERANCE_DEG:
                turn = direction_phrase(rel).replace("behind you, to the", "around to the").replace("behind you", "around")
                return self._guidance(f"Turn {turn} to face your route.", f"face:{'r' if rel > 0 else 'l'}:{idx}",
                                      remaining_route, rel, idx, maneuver="face")
        if step["type"] == "arrive":
            text = f"Continue for {round_distance(to_turn)} to reach {r['destination_name']}."
            key = f"arrive:{int(to_turn // 25)}"
        elif to_turn <= TURN_NOW_M:
            phrase = maneuver_phrase(step)
            text, key = f"Now, {phrase}.", f"step{idx}:now"
        elif to_turn <= TURN_SOON_M:
            phrase = maneuver_phrase(step)
            text, key = f"In {round_distance(to_turn)}, {phrase}.", f"step{idx}:soon"
        else:
            phrase = maneuver_phrase(step)
            text, key = f"Continue straight for {round_distance(to_turn)}, then {phrase}.", f"step{idx}:far"
        return self._guidance(text, key, remaining_route, rel, idx, maneuver=step.get("maneuver", "straight"),
                              urgent=key.endswith(":now"), to_turn=to_turn)

    def _guidance(self, text, key, remaining, rel, idx, maneuver="straight", urgent=False, to_turn=None):
        r = self.active_route
        return {
            "status": "navigating",
            "destination": r["destination_name"] if r else self.destination_name,
            "instruction": text,
            "announce_key": key,
            "urgent": urgent,
            "dist_remaining_m": round(remaining, 1),
            "dist_to_turn_m": None if to_turn is None else round(to_turn, 1),
            "relative_deg": None if rel is None else round(rel),
            "step_index": idx + 1,
            "maneuver": maneuver,
            "mode": r["mode"] if r else "direct",
            "is_completed": False,
        }

    def describe_position(self, lat, lon, heading=None, accuracy=None):
        """
        'Where am I?': the street and area (from the map, when online) and the nearest saved place or
        campus landmark within 300 m, told by clock direction when the facing is known.
        """
        parts = []
        if self.online:
            try:
                where = self._reverse(lat, lon)
            except Exception as e:                              # offline or service busy
                print(f"[Navigation] Address lookup failed: {e}")
                where = None
            if where:
                road, area, city = where.get("road"), where.get("area"), where.get("city")
                place = ", ".join(x for x in (area, city) if x)
                if road:
                    parts.append(f"You are on {road}" + (f", in {place}." if place else "."))
                elif place:
                    parts.append(f"You are in {place}.")
        known = [dict(p, _m=haversine_m(lat, lon, p["lat"], p["lon"])) for p in self.places.all()]
        known += [dict(wp, _m=haversine_m(lat, lon, wp["lat"], wp["lon"])) for wp in SKIT_WAYPOINTS.values()]
        near = min(known, key=lambda p: p["_m"]) if known else None
        if near and near["_m"] <= 300:
            if near["_m"] <= max(ARRIVE_RADIUS_M, min(20.0, accuracy or 0.0)):
                parts.append(f"You are at {near['name']}.")
            else:
                b = bearing_deg(lat, lon, near["lat"], near["lon"])
                way = (f"at {clock_position(relative_angle(b, heading))} o'clock" if heading is not None
                       else f"to the {COMPASS_WORDS[int((b + 22.5) // 45) % 8]}")
                parts.append(f"{near['name']} is {round_distance(near['_m'])} away, {way}.")
        if not parts:
            parts.append("I know your position, but I cannot name this place. Save it so I can guide you back here.")
        if accuracy and accuracy > 50:
            parts.append(f"Location is only accurate to about {round_distance(accuracy)}.")
        return " ".join(parts)

    def cancel_navigation(self):
        """Stops active wayfinding."""
        self.active_route = None
        self.current_step_idx = 0
        self.destination_name = None
        return {"status": "cancelled", "message": "Navigation cancelled."}
