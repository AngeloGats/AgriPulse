"""Focused tests for the KNN demo routes."""

from __future__ import annotations

import unittest

import pandas as pd

from app import app
from knn import DATASETS, MAX_K, DATA_DIR, MODEL_DIR

import joblib


class KnnDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def _get(self, path: str):
        response = self.client.get(path)
        self.addCleanup(response.close)
        return response

    def _post(self, path: str, payload):
        response = self.client.post(path, json=payload)
        self.addCleanup(response.close)
        return response

    def test_page_and_assets_load(self) -> None:
        page = self._get("/knn")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"K-Nearest Neighbors", page.data)
        self.assertIn(b"How does KNN reach an answer?", page.data)
        for asset in ("knn.css", "knn.js"):
            self.assertEqual(self._get(f"/ml-static/{asset}").status_code, 200, asset)

    def test_chart_data_for_every_dataset(self) -> None:
        for name in DATASETS:
            data = self._get(f"/api/knn/{name}/chart").get_json()
            self.assertEqual(len(data["accuracy_by_k"]), MAX_K)
            self.assertTrue(data["points"])
            self.assertLessEqual(sum(data["explained_variance"]), 1.0)
            self.assertTrue(all(0 <= p[2] < len(data["classes"]) for p in data["points"]))
        self.assertEqual(self._get("/api/knn/nope/chart").status_code, 404)

    def test_server_prediction_matches_trained_model(self) -> None:
        for name in DATASETS:
            bundle = joblib.load(MODEL_DIR / f"{name}.joblib")
            test = pd.read_csv(DATA_DIR / name / "test.csv").head(40)
            for _, row in test.iterrows():
                inputs = {f: float(row[f]) for f in bundle["features"]}
                data = self._post(f"/api/knn/{name}/predict", {"inputs": inputs}).get_json()
                expected = bundle["pipeline"].predict(pd.DataFrame([inputs]))[0]
                self.assertEqual(data["prediction_key"], expected, f"{name} sample {row['sample_id']}")

    def test_votes_and_neighbors_are_consistent(self) -> None:
        example = self._get("/api/knn/soil_type/example?n=3").get_json()
        data = self._post("/api/knn/soil_type/predict", {"inputs": example["inputs"], "k": 7}).get_json()
        self.assertEqual(data["k"], 7)
        self.assertEqual(sum(v["count"] for v in data["votes"]), 7)
        self.assertAlmostEqual(sum(v["share"] for v in data["votes"]), 1.0, places=6)
        self.assertEqual(len(data["by_k"]), MAX_K)
        self.assertEqual(data["by_k"][6]["prediction"], data["prediction"])
        distances = [n["distance"] for n in data["neighbors"]]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual(len(data["neighbors"]), MAX_K)

    def test_k_one_follows_single_nearest_neighbor(self) -> None:
        example = self._get("/api/knn/plant_condition/example").get_json()
        data = self._post("/api/knn/plant_condition/predict", {"inputs": example["inputs"], "k": 1}).get_json()
        self.assertEqual(data["prediction"], data["neighbors"][0]["label"])

    def test_invalid_requests_are_rejected(self) -> None:
        example = self._get("/api/knn/soil_fertility/example").get_json()["inputs"]
        url = "/api/knn/soil_fertility/predict"
        cases = [
            {"inputs": {**example, "ph": 20}},
            {"inputs": {**example, "ph": "abc"}},
            {"inputs": {**example, "ph": True}},
            {"inputs": {**example, "extra": 1}},
            {"inputs": {k: v for k, v in example.items() if k != "ph"}},
            {"inputs": example, "k": 0},
            {"inputs": example, "k": 26},
            {"inputs": example, "k": 3.5},
            {"inputs": example, "k": True},
            {"inputs": [1, 2]},
            {"nothing": 1},
        ]
        for payload in cases:
            self.assertEqual(self._post(url, payload).status_code, 400, payload)
        self.assertEqual(self._post("/api/knn/nope/predict", {"inputs": example}).status_code, 404)
        not_json = self.client.post(url, data="x", content_type="text/plain")
        self.addCleanup(not_json.close)
        self.assertEqual(not_json.status_code, 400)

    def test_far_away_sample_gets_a_warning(self) -> None:
        example = self._get("/api/knn/soil_fertility/example").get_json()["inputs"]
        data = self._post("/api/knn/soil_fertility/predict", {"inputs": {**example, "nitrogen_kg_ha": 900}}).get_json()
        self.assertTrue(data["warnings"])

    def test_texture_not_adding_to_100_warns(self) -> None:
        example = self._get("/api/knn/soil_type/example").get_json()["inputs"]
        data = self._post("/api/knn/soil_type/predict", {"inputs": {**example, "sand_pct": 90, "silt_pct": 40, "clay_pct": 40}}).get_json()
        self.assertTrue(any("add up" in w for w in data["warnings"]))


if __name__ == "__main__":
    unittest.main()
