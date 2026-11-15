"""
Unit Tests for VivaEvaluationSandbox (Examiner Viva Evaluation Sandbox).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
from modules.viva_evaluation_sandbox import VivaEvaluationSandbox

class TestVivaEvaluationSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = VivaEvaluationSandbox()

    def test_available_scenarios(self):
        scenarios = self.sandbox.get_available_scenarios()
        self.assertGreaterEqual(len(scenarios), 5)
        titles = [s["title"] for s in scenarios]
        self.assertTrue(any("TTC" in t for t in titles))
        self.assertTrue(any("Medicine" in t for t in titles))

    def test_run_benchmark_suite(self):
        metrics = self.sandbox.run_benchmark_suite()
        self.assertIn("total_vision_latency_ms", metrics)
        self.assertLess(metrics["total_vision_latency_ms"], 100.0)
        self.assertGreater(metrics["effective_fps"], 15.0)
        self.assertGreater(metrics["detection_accuracy_pct"], 90.0)

    def test_execute_scenario(self):
        res = self.sandbox.execute_scenario("speeding_vehicle_ttc")
        self.assertEqual(res["status"], "passed")
        self.assertTrue(res["verified"])
        self.assertIn("evasive", res["alert_dispatched"].lower())


class TestSandboxAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_sandbox_page_render(self):
        res = self.client.get("/sandbox")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("EXAMINER VIVA EVALUATION", html)
        self.assertIn("Amit Tiwari", html)
        self.assertIn("Vikas", html)
        self.assertIn("Scenario A: High-Speed Approaching Vehicle", html)

    def test_api_benchmark(self):
        res = self.client.get("/api/sandbox/benchmark")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("benchmark", data)
        self.assertIn("effective_fps", data["benchmark"])

    def test_api_simulate(self):
        res = self.client.post("/api/sandbox/simulate", json={"scenario_id": "speeding_vehicle_ttc"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "passed")
        self.assertTrue(data["verified"])


if __name__ == "__main__":
    unittest.main()
