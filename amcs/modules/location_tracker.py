"""
Location Tracker & Folium Mapping Module.
Provides real-time GPS coordinate logging, breadcrumb paths, geofencing,
and interactive Folium live map generation for caregivers.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import time
import math
import folium
from folium import plugins
import config

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class LocationTracker:
    def __init__(self):
        # Default starting point: SKIT Jaipur campus
        self.current_lat = config.DEFAULT_LATITUDE
        self.current_lon = config.DEFAULT_LONGITUDE
        self.current_accuracy = 10.0  # meters
        self.location_name = config.DEFAULT_LOCATION_LABEL
        self.last_updated = time.time()
        
        # Path history (list of {"lat": ..., "lon": ..., "time": ...})
        self.history = [
            {"lat": self.current_lat, "lon": self.current_lon, "time": time.time()}
        ]

        # Safe Geofence Center (SKIT Campus)
        self.safe_center_lat = config.DEFAULT_LATITUDE
        self.safe_center_lon = config.DEFAULT_LONGITUDE
        self.geofence_radius = config.GEOFENCE_RADIUS_METERS

        # Emergency SOS Alert state
        self.sos_active = False
        self.sos_message = ""
        self.sos_timestamp = None

    def update_location(self, lat, lon, accuracy=None, location_name=None):
        """Updates current GPS position and records breadcrumb trail."""
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return False

        self.current_lat = lat
        self.current_lon = lon
        if accuracy is not None:
            self.current_accuracy = float(accuracy)
        if location_name:
            self.location_name = location_name

        self.last_updated = time.time()

        # Append to breadcrumb trail if moved significantly (> 3 meters) or first point
        if not self.history:
            self.history.append({"lat": lat, "lon": lon, "time": self.last_updated})
        else:
            last = self.history[-1]
            dist = haversine_distance(last["lat"], last["lon"], lat, lon)
            if dist >= 3.0:
                self.history.append({"lat": lat, "lon": lon, "time": self.last_updated})
                # Keep max 150 points for clean rendering
                if len(self.history) > 150:
                    self.history.pop(0)

        return True

    def trigger_sos(self, message="User triggered Emergency SOS Alert!"):
        """Triggers emergency caregiver alert."""
        self.sos_active = True
        self.sos_message = message
        self.sos_timestamp = time.time()

    def resolve_sos(self):
        """Clears emergency SOS status."""
        self.sos_active = False
        self.sos_message = ""
        self.sos_timestamp = None

    def calculate_total_distance_m(self):
        """Calculates total accumulated distance walked in meters along breadcrumb history."""
        if len(self.history) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.history)):
            p1 = self.history[i - 1]
            p2 = self.history[i]
            total += haversine_distance(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        return round(total, 1)

    def get_safety_analytics(self):
        """Generates comprehensive mobility summary for caregiver reporting."""
        total_m = self.calculate_total_distance_m()
        duration_s = max(0, time.time() - self.history[0]["time"]) if self.history else 0
        minutes = round(duration_s / 60.0, 1)
        geofence = self.check_geofence()

        # Compute dynamic safety score (100 is optimal safe navigation)
        safety_score = 100
        if geofence["breached"]:
            safety_score -= 30
        if self.sos_active:
            safety_score -= 50
        safety_score = max(0, min(100, safety_score))

        return {
            "safety_score": safety_score,
            "total_distance_m": total_m,
            "total_distance_km": round(total_m / 1000.0, 2),
            "duration_minutes": minutes,
            "waypoints_recorded": len(self.history),
            "safe_zone_radius_m": self.geofence_radius,
            "current_distance_from_center_m": geofence["distance_from_center_m"],
            "geofence_breached": geofence["breached"],
            "sos_active": self.sos_active,
            "last_location": f"{round(self.current_lat, 4)}° N, {round(self.current_lon, 4)}° E",
            "start_location": f"{round(self.history[0]['lat'], 4)}° N, {round(self.history[0]['lon'], 4)}° E" if self.history else "N/A"
        }

    def check_geofence(self):
        """Checks if user has drifted outside configured safety perimeter."""
        dist = haversine_distance(self.current_lat, self.current_lon,
                                  self.safe_center_lat, self.safe_center_lon)
        is_breached = dist > self.geofence_radius
        return {
            "breached": is_breached,
            "distance_from_center_m": round(dist, 1),
            "safe_radius_m": self.geofence_radius
        }

    def generate_folium_map(self):
        """
        Builds a modern, interactive Folium Leaflet map with user position,
        breadcrumbs, geofence circle, and emergency indicators.
        Returns HTML string suitable for embedding.
        """
        # Create base map using standard OpenStreetMap (no API key required)
        m = folium.Map(
            location=[self.current_lat, self.current_lon],
            zoom_start=config.DEFAULT_MAP_ZOOM,
            tiles="OpenStreetMap",
            control_scale=True
        )

        # 1. Geofence Safe Zone Circle (Glowing Emerald)
        folium.Circle(
            location=[self.safe_center_lat, self.safe_center_lon],
            radius=self.geofence_radius,
            color="#10b981",
            weight=2,
            fill=True,
            fill_color="#10b981",
            fill_opacity=0.08,
            dash_array="6, 6",
            popup=f"Designated Safe Zone ({self.geofence_radius}m radius)"
        ).add_to(m)

        # 2. Breadcrumb Travel Trail (Glowing Cyan Polyline)
        if len(self.history) >= 2:
            points = [[p["lat"], p["lon"]] for p in self.history]
            folium.PolyLine(
                locations=points,
                color="#38bdf8",
                weight=4.5,
                opacity=0.9,
                dash_array="4, 6",
                tooltip="Live Movement Trail"
            ).add_to(m)

            # Start point marker
            folium.CircleMarker(
                location=[points[0][0], points[0][1]],
                radius=6,
                color="#64748b",
                fill=True,
                fill_color="#38bdf8",
                fill_opacity=0.9,
                popup="Journey Starting Point"
            ).add_to(m)

        # 3. Accuracy Radius Circle
        folium.Circle(
            location=[self.current_lat, self.current_lon],
            radius=max(5, self.current_accuracy),
            color="#ef4444" if self.sos_active else "#6366f1",
            fill=True,
            fill_color="#ef4444" if self.sos_active else "#6366f1",
            fill_opacity=0.18
        ).add_to(m)

        # 4. Current User Location Marker with Pulse
        marker_color = "red" if self.sos_active else "blue"
        marker_icon = "exclamation-triangle" if self.sos_active else "user"
        
        status_text = "EMERGENCY SOS ACTIVE" if self.sos_active else "Active Telemetry"
        popup_html = f"""
        <div style="font-family:'Plus Jakarta Sans',sans-serif; min-width:190px; padding:6px; color:#f1f2f6;">
            <b style="color:{'#f87171' if self.sos_active else '#38bdf8'}; font-size:13px; text-transform:uppercase; letter-spacing:0.04em;">
                {status_text}
            </b>
            <p style="margin:5px 0 2px 0; font-size:12px; color:#e2e8f0;"><b>Campus:</b> {self.location_name}</p>
            <p style="margin:2px 0; font-size:11px; color:#94a3b8; font-family:monospace;">{round(self.current_lat, 5)}° N, {round(self.current_lon, 5)}° E</p>
            <p style="margin:2px 0; font-size:11px; color:#64748b;">Accuracy: ±{round(self.current_accuracy, 1)}m</p>
        </div>
        """

        folium.Marker(
            location=[self.current_lat, self.current_lon],
            popup=folium.Popup(popup_html, max_width=270),
            tooltip="User Current Location",
            icon=folium.Icon(color=marker_color, icon=marker_icon, prefix="fa")
        ).add_to(m)

        folium.LayerControl(position="topright").add_to(m)

        dark_css = """
        <style>
            .leaflet-container { background: #08090d !important; font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; }
            .leaflet-tile-pane { filter: brightness(0.65) invert(1) contrast(2.8) hue-rotate(200deg) saturate(0.22) brightness(0.75); }
            .leaflet-popup-content-wrapper { background: rgba(15, 17, 24, 0.95) !important; color: #f1f2f6 !important; backdrop-filter: blur(16px) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 14px !important; box-shadow: 0 12px 36px rgba(0,0,0,0.6) !important; }
            .leaflet-popup-tip { background: rgba(15, 17, 24, 0.95) !important; }
            .leaflet-bar a { background-color: rgba(20, 22, 33, 0.9) !important; color: #f1f2f6 !important; border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
            .leaflet-bar a:hover { background-color: #6366f1 !important; color: #fff !important; }
        </style>
        """

        # Return embeddable HTML string with Obsidian dark styles
        return dark_css + m._repr_html_()

    def get_status_payload(self):
        """Returns JSON serializable status dictionary for caregiver polling."""
        geofence = self.check_geofence()
        return {
            "latitude": self.current_lat,
            "longitude": self.current_lon,
            "accuracy": self.current_accuracy,
            "location_name": self.location_name,
            "last_updated": round(self.last_updated, 2),
            "history_count": len(self.history),
            "sos_active": self.sos_active,
            "sos_message": self.sos_message,
            "geofence": geofence
        }
