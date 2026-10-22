"""Focused tests for the local Linear Regression demo."""

from __future__ import annotations

import json
import re
import unittest

from app import app


class LinearRegressionPageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self.example = self._get("/api/example").get_json()

    def _get(self, path: str):
        response = self.client.get(path)
        self.addCleanup(response.close)
        return response

    def _post(self, path: str, **kwargs):
        response = self.client.post(path, **kwargs)
        self.addCleanup(response.close)
        return response

    def test_page_and_local_assets_load(self) -> None:
        page = self._get("/linear-regression")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Estimate crop yield", page.data)
        self.assertIn(b"Calculate yield estimate", page.data)
        self.assertIn(b"Actual vs. predicted yield", page.data)
        self.assertIn(b"Your estimate", page.data)
        self.assertIn(b"What do these inputs mean for your field?", page.data)
        self.assertIn(b"Growth Variability", page.data)
        self.assertIn(b"Seasonal Average", page.data)
        self.assertIn(b"Canopy &amp; Biomass Return", page.data)
        self.assertIn(b'id="regression-chart"', page.data)
        self.assertIn(b'id="regression-chart-data"', page.data)
        chart_data = re.search(
            rb'<script id="regression-chart-data" type="application/json">(.*?)</script>',
            page.data,
            re.DOTALL,
        )
        self.assertIsNotNone(chart_data)
        points = json.loads(chart_data.group(1))
        self.assertEqual(len(points), 192)
        self.assertEqual(set(points[0]), {"actual", "predicted"})
        self.assertEqual(
            self._get("/ml-static/linear-regression.js").status_code,
            200,
        )
        self.assertEqual(
            self._get("/ml-static/linear-regression.css").status_code,
            200,
        )

    def test_existing_algorithm_library_is_served_by_local_app(self) -> None:
        page = self._get("/analytics.html")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Try the trained Linear Regression model", page.data)
        self.assertIn(b'class="model-demo-button" href="/linear-regression"', page.data)
        self.assertIn(b'href="/linear-regression"', page.data)
        self.assertIn(b"data-model-demo", page.data)
        self.assertIn(b'aria-label="Open the Linear Regression model"', page.data)
        self.assertEqual(
            self._get("/assets/js/script.js").status_code,
            200,
        )
        self.assertEqual(
            self._get("/assets/css/styles.css").status_code,
            200,
        )

    def test_site_homepage_is_the_root_route(self) -> None:
        page = self._get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"<title>AgroPulse Intelligence", page.data)
        self.assertIn(b"Turn field data into clearer decisions", page.data)

    def test_all_site_pages_and_assets_are_served(self) -> None:
        for page in ("index", "about", "analytics", "demo"):
            with self.subTest(page=page):
                response = self._get(f"/{page}.html")
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"../assets/", response.data)

        self.assertEqual(
            self._get("/assets/images/Farmer1.jpg").status_code,
            200,
        )
        self.assertEqual(
            self._get("/assets/js/home.js").status_code,
            200,
        )

    def test_example_runs_through_the_trained_model(self) -> None:
        response = self._post("/api/predict", json=self.example)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["crop"], self.example["crop"])
        self.assertEqual(result["year"], self.example["year"])
        self.assertIn("prediction_t_per_ha", result)
        self.assertEqual(result["model_metrics"]["test_rows"], 192)
        self.assertNotIn("sensor_summaries", result)

    def test_invalid_satellite_summary_is_rejected(self) -> None:
        invalid = {
            **self.example,
            "ratio": {
                **self.example["ratio"],
                "mean": self.example["ratio"]["max"] + 1,
            },
        }
        response = self._post("/api/predict", json=invalid)
        self.assertEqual(response.status_code, 400)
        self.assertIn("min <= mean <= max", response.get_json()["error"])

    def test_zero_observations_use_training_medians(self) -> None:
        no_readings = {
            **self.example,
            "ratio": {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "observations": 0,
            },
        }
        response = self._post("/api/predict", json=no_readings)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn("prediction_t_per_ha", result)

    def test_non_json_request_is_rejected(self) -> None:
        response = self._post("/api/predict", data="not json")
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
