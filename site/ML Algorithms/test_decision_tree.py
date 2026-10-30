"""Focused tests for the Decision Tree demo routes."""

from __future__ import annotations

import unittest

import joblib
import pandas as pd

from app import app
from decision_tree import DATA_DIR, MODEL_PATH


class DecisionTreeDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def _get(self, path: str):
        response = self.client.get(path)
        self.addCleanup(response.close)
        return response

    def _post(self, payload):
        response = self.client.post("/api/dt/predict", json=payload)
        self.addCleanup(response.close)
        return response

    INPUTS = {"soil_moisture_pct": 18, "rain_last_3d_mm": 0, "forecast_rain_mm": 0, "crop_stage": 3, "air_temp_c": 30}

    def test_page_and_assets_load(self) -> None:
        page = self._get("/decision-tree")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Decision Tree", page.data)
        self.assertIn(b"How does a decision tree reach an answer?", page.data)
        for asset in ("decision-tree.css", "decision-tree.js"):
            self.assertEqual(self._get(f"/ml-static/{asset}").status_code, 200, asset)

    def test_tree_structure_is_consistent(self) -> None:
        for depth in range(1, 6):
            data = self._get(f"/api/dt/tree?depth={depth}").get_json()
            nodes = {n["id"]: n for n in data["nodes"]}
            leaves = [n for n in nodes.values() if n["left"] is None]
            self.assertEqual(len(leaves), data["leaves"])
            for node in nodes.values():
                if node["left"] is not None:
                    children = nodes[node["left"]]["samples"] + nodes[node["right"]]["samples"]
                    self.assertEqual(children, node["samples"])
                    self.assertTrue(node["question"].endswith("?"))
            self.assertEqual(nodes[0]["samples"], 1280)
        self.assertEqual(self._get("/api/dt/tree?depth=9").status_code, 400)
        self.assertEqual(self._get("/api/dt/tree?depth=abc").status_code, 400)

    def test_server_matches_trained_model(self) -> None:
        bundle = joblib.load(MODEL_PATH)
        test = pd.read_csv(DATA_DIR / "test.csv").head(60)
        for _, row in test.iterrows():
            inputs = {f: float(row[f]) for f in bundle["features"]}
            data = self._post({"inputs": inputs}).get_json()
            expected = bundle["model"].predict(pd.DataFrame([inputs]))[0]
            self.assertEqual(data["prediction"], expected, row["sample_id"])

    def test_path_and_rules_are_consistent(self) -> None:
        data = self._post({"inputs": self.INPUTS, "depth": 4}).get_json()
        self.assertEqual(data["path"][0], 0)
        self.assertEqual(data["path"][-1], data["leaf"])
        self.assertEqual(len(data["steps"]), len(data["path"]) - 1)
        self.assertEqual([d["depth"] for d in data["by_depth"]], [1, 2, 3, 4, 5])
        self.assertEqual(sum(c["count"] for c in data["leaf_counts"]), data["leaf_samples"])
        self.assertTrue(0 < data["leaf_purity"] <= 1)

    def test_dry_flowering_field_is_watered_and_wet_field_is_held(self) -> None:
        dry = self._post({"inputs": self.INPUTS}).get_json()
        self.assertEqual(dry["prediction"], "irrigate_now")
        wet = self._post({"inputs": {**self.INPUTS, "soil_moisture_pct": 42}}).get_json()
        self.assertEqual(wet["prediction"], "hold")

    def test_example_endpoint(self) -> None:
        data = self._get("/api/dt/example?n=2").get_json()
        self.assertEqual(set(data["inputs"]), set(self.INPUTS))
        self.assertIn(data["actual"], ("Irrigate now", "Irrigate soon", "Wait", "Hold off"))

    def test_far_away_value_warns(self) -> None:
        data = self._post({"inputs": {**self.INPUTS, "air_temp_c": 58}}).get_json()
        self.assertTrue(data["warnings"])

    def test_invalid_requests_are_rejected(self) -> None:
        cases = [
            {"inputs": {**self.INPUTS, "soil_moisture_pct": 150}},
            {"inputs": {**self.INPUTS, "soil_moisture_pct": "abc"}},
            {"inputs": {**self.INPUTS, "soil_moisture_pct": True}},
            {"inputs": {**self.INPUTS, "crop_stage": 2.5}},
            {"inputs": {**self.INPUTS, "crop_stage": 7}},
            {"inputs": {**self.INPUTS, "extra": 1}},
            {"inputs": {k: v for k, v in self.INPUTS.items() if k != "air_temp_c"}},
            {"inputs": self.INPUTS, "depth": 0},
            {"inputs": self.INPUTS, "depth": 6},
            {"inputs": self.INPUTS, "depth": 2.5},
            {"inputs": self.INPUTS, "depth": True},
            {"nothing": 1},
            [1, 2],
        ]
        for payload in cases:
            self.assertEqual(self._post(payload).status_code, 400, payload)
        not_json = self.client.post("/api/dt/predict", data="x", content_type="text/plain")
        self.addCleanup(not_json.close)
        self.assertEqual(not_json.status_code, 400)


if __name__ == "__main__":
    unittest.main()
