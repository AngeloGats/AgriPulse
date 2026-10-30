"""Focused tests for the SVM demo routes."""

from __future__ import annotations

import unittest

from app import app


class SvmDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def _get(self, path: str):
        response = self.client.get(path)
        self.addCleanup(response.close)
        return response

    def _post(self, payload):
        response = self.client.post("/api/svm/model", json=payload)
        self.addCleanup(response.close)
        return response

    BASE = {"x": "air_temp_c", "y": "humidity_pct", "kernel": "rbf", "c": 1, "gamma": 0.5}

    def test_page_and_assets_load(self) -> None:
        page = self._get("/svm")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Support Vector Machine", page.data)
        self.assertIn(b"How does an SVM reach an answer?", page.data)
        for asset in ("svm.css", "svm.js"):
            self.assertEqual(self._get(f"/ml-static/{asset}").status_code, 200, asset)

    def test_model_response_shape(self) -> None:
        data = self._post({**self.BASE, "sample": {"air_temp_c": 24, "humidity_pct": 85}}).get_json()
        self.assertEqual(len(data["grid"]), data["grid_size"] ** 2)
        self.assertEqual(len(data["points"]), data["train_rows"])
        self.assertEqual(sum(p[3] for p in data["points"]), data["support_vectors"])
        self.assertTrue(0 < data["test_accuracy"] <= 1)
        sample = data["sample"]
        self.assertEqual(sample["prediction"], "Diseased" if sample["score"] > 0 else "Healthy")
        self.assertEqual(sample["in_margin"], abs(sample["score"]) < 1)

    def test_curved_kernel_beats_straight_line_on_temperature(self) -> None:
        linear = self._post({**self.BASE, "kernel": "linear", "gamma": None}).get_json()
        curved = self._post(self.BASE).get_json()
        self.assertGreater(curved["test_accuracy"], linear["test_accuracy"] + 0.05)

    def test_larger_c_means_fewer_support_vectors(self) -> None:
        loose = self._post({**self.BASE, "kernel": "linear", "gamma": None, "c": 0.01}).get_json()
        strict = self._post({**self.BASE, "kernel": "linear", "gamma": None, "c": 100}).get_json()
        self.assertGreater(loose["support_vectors"], strict["support_vectors"])

    def test_diseased_zone_is_scored_diseased(self) -> None:
        data = self._post({**self.BASE, "sample": {"air_temp_c": 24, "humidity_pct": 88}}).get_json()
        self.assertEqual(data["sample"]["prediction"], "Diseased")
        cool_dry = self._post({**self.BASE, "sample": {"air_temp_c": 24, "humidity_pct": 40}}).get_json()
        self.assertEqual(cool_dry["sample"]["prediction"], "Healthy")

    def test_example_endpoint(self) -> None:
        data = self._get("/api/svm/example?n=2").get_json()
        self.assertEqual(len(data["inputs"]), 4)
        self.assertIn(data["actual"], ("Healthy", "Diseased"))

    def test_invalid_requests_are_rejected(self) -> None:
        sample = {"air_temp_c": 24, "humidity_pct": 80}
        cases = [
            {**self.BASE, "x": "nope"},
            {**self.BASE, "y": "air_temp_c"},
            {**self.BASE, "kernel": "poly"},
            {**self.BASE, "c": 0},
            {**self.BASE, "c": 1000},
            {**self.BASE, "c": True},
            {**self.BASE, "c": "abc"},
            {**self.BASE, "gamma": 0},
            {**self.BASE, "gamma": None},
            {**self.BASE, "sample": {"air_temp_c": 24}},
            {**self.BASE, "sample": {**sample, "humidity_pct": 150}},
            {**self.BASE, "sample": {**sample, "extra": 1}},
            {**self.BASE, "sample": {"air_temp_c": "x", "humidity_pct": 5}},
            [1, 2],
        ]
        for payload in cases:
            self.assertEqual(self._post(payload).status_code, 400, payload)
        not_json = self.client.post("/api/svm/model", data="x", content_type="text/plain")
        self.addCleanup(not_json.close)
        self.assertEqual(not_json.status_code, 400)


if __name__ == "__main__":
    unittest.main()
