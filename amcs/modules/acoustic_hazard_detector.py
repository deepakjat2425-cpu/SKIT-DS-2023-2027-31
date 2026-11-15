"""
Acoustic Siren & Car Horn AI Detector Engine (Multimodal Audio Sensor Fusion).
Monitors environmental audio for hazards outside the camera's field of view
such as speeding vehicles honking behind the user, ambulance sirens, or barking dogs.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import numpy as np

class AcousticHazardDetector:
    def __init__(self):
        self.sample_rate = 16000
        # Energy thresholds
        self.horn_min_energy = 0.08
        self.siren_min_energy = 0.06

    def analyze_audio_spectrum(self, frequency_bins, sample_rate=16000):
        """
        Analyzes frequency bin magnitudes from Web Audio FFT Analyser.
        frequency_bins: list of float magnitudes (0-255 or normalized 0.0-1.0)
        Returns:
            dict: { detected: bool, hazard_type: str, confidence: float, severity: str, advisory_en: str }
        """
        if not frequency_bins or len(frequency_bins) < 64:
            return {"detected": False, "hazard_type": "none", "confidence": 0.0, "severity": "none", "advisory_en": ""}

        bins = np.array(frequency_bins, dtype=np.float32)
        if np.max(bins) > 1.0:
            bins = bins / 255.0  # Normalize to 0.0 - 1.0

        n_bins = len(bins)
        hz_per_bin = (sample_rate / 2.0) / n_bins

        # 1. Automotive Horn Detection: Strong energy concentration around 2800 Hz - 4500 Hz
        horn_bin_start = int(2500 / hz_per_bin)
        horn_bin_end = min(n_bins, int(4500 / hz_per_bin))
        horn_energy = np.mean(bins[horn_bin_start:horn_bin_end]) if horn_bin_end > horn_bin_start else 0.0

        # Baseline noise (low & mid frequencies)
        baseline_energy = np.mean(bins[:horn_bin_start]) if horn_bin_start > 0 else 0.01

        # 2. Emergency Siren Detection: Modulated energy peak in 700 Hz - 1600 Hz
        siren_bin_start = int(650 / hz_per_bin)
        siren_bin_end = int(1600 / hz_per_bin)
        siren_energy = np.mean(bins[siren_bin_start:siren_bin_end]) if siren_bin_end > siren_bin_start else 0.0

        # Ratio checks to avoid false alarms from human speech or ambient room hum
        horn_ratio = horn_energy / (baseline_energy + 1e-4)
        siren_ratio = siren_energy / (baseline_energy + 1e-4)

        if horn_energy >= 0.40 and horn_ratio >= 1.6:
            conf = min(0.98, round(float(horn_energy * 1.2), 2))
            return {
                "detected": True,
                "hazard_type": "vehicle_horn",
                "confidence": conf,
                "severity": "hazard",
                "advisory_en": "Caution! Loud vehicle horn detected nearby! Move to the side of the path."
            }

        if siren_energy >= 0.35 and siren_ratio >= 1.5:
            conf = min(0.95, round(float(siren_energy * 1.15), 2))
            return {
                "detected": True,
                "hazard_type": "emergency_siren",
                "confidence": conf,
                "severity": "hazard",
                "advisory_en": "Emergency siren detected approaching! Clear the roadway."
            }

        return {
            "detected": False,
            "hazard_type": "none",
            "confidence": 0.0,
            "severity": "none",
            "advisory_en": ""
        }

    def simulate_test_signal(self, hazard_type="horn", n_bins=128, sample_rate=16000):
        """Generates synthetic frequency bin distributions for automated unit testing."""
        bins = np.ones(n_bins, dtype=np.float32) * 0.05
        hz_per_bin = (sample_rate / 2.0) / n_bins

        if hazard_type == "horn":
            b_start = int(2800 / hz_per_bin)
            b_end = int(4200 / hz_per_bin)
            bins[b_start:b_end] = 0.75
        elif hazard_type == "siren":
            b_start = int(800 / hz_per_bin)
            b_end = int(1400 / hz_per_bin)
            bins[b_start:b_end] = 0.70

        return [float(x) for x in bins]
