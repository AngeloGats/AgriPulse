"""Focused tests for the Naive Bayes demo routes."""

from __future__ import annotations

import unittest

from app import app


class NaiveBayesDemoTests(unittest.TestCase):
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
        page = self._get("/naive-bayes")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Diagnose and recommend", page.data)
        self.assertIn(b"Soybean disease diagnosis", page.data)
        self.assertIn(b"Crop recommendation", page.data)
        self.assertIn(b"How does Naive Bayes reach an answer?", page.data)
        self.assertEqual(page.data.count(b"<select"), 35)
        for asset in ("naive-bayes.css", "naive-bayes.js"):
            response = self._get(f"/ml-static/{asset}")
            self.assertEqual(response.status_code, 200, asset)

    def test_soybean_example_predicts_with_valid_probabilities(self) -> None:
        example = self._get("/api/nb/soybean/example?n=0").get_json()
        self.assertTrue(example["symptoms"])
        response = self._post("/api/nb/soybean/predict", {"symptoms": example["symptoms"]})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        probabilities = [item["probability"] for item in data["predictions"]]
        self.assertEqual(len(probabilities), 15)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertEqual(data["top"], data["predictions"][0]["name"])
        self.assertEqual(data["symptoms_used"], len(example["symptoms"]))
        self.assertTrue(data["evidence"])

    def test_soybean_examples_are_mostly_diagnosed_correctly(self) -> None:
        correct = 0
        for index in range(40):
            example = self._get(f"/api/nb/soybean/example?n={index}").get_json()
            data = self._post("/api/nb/soybean/predict", {"symptoms": example["symptoms"]}).get_json()
            correct += data["top"] == example["actual"]
        self.assertGreaterEqual(correct / 40, 0.8)

    def test_soybean_skipped_symptoms_are_ignored(self) -> None:
        example = self._get("/api/nb/soybean/example?n=0").get_json()
        key, code = next(iter(example["symptoms"].items()))
        base = self._post("/api/nb/soybean/predict", {"symptoms": {key: code}}).get_json()
        with_blank = self._post(
            "/api/nb/soybean/predict", {"symptoms": {key: code, "stem": None, "roots": ""}}
        ).get_json()
        self.assertEqual(base["predictions"], with_blank["predictions"])
        self.assertEqual(with_blank["symptoms_used"], 1)

    def test_soybean_rejects_bad_input(self) -> None:
        cases = [
            {"symptoms": {}},
            {"symptoms": {"stem": None}},
            {"symptoms": {"not-a-symptom": 0}},
            {"symptoms": {"stem": 99}},
            {"symptoms": {"stem": -1}},
            {"symptoms": {"stem": "0"}},
            {"symptoms": {"stem": True}},
            {"symptoms": [1, 2]},
            [],
        ]
        for payload in cases:
            response = self._post("/api/nb/soybean/predict", payload)
            self.assertEqual(response.status_code, 400, payload)
            self.assertIn("error", response.get_json())

    def test_crop_example_prediction_and_evidence(self) -> None:
        example = self._get("/api/nb/crop/example?n=0").get_json()
        response = self._post("/api/nb/crop/predict", {"inputs": example["inputs"]})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        probabilities = [item["probability"] for item in data["predictions"]]
        self.assertEqual(len(probabilities), 22)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertEqual(len(data["evidence"]), 7)
        self.assertEqual(data["top"], example["actual"])

    def test_crop_flags_values_outside_training_range(self) -> None:
        inputs = self._get("/api/nb/crop/example?n=0").get_json()["inputs"]
        inputs["rainfall"] = 900
        data = self._post("/api/nb/crop/predict", {"inputs": inputs}).get_json()
        self.assertIn("Rainfall", data["outside_training_range"])

    def test_crop_rejects_bad_input(self) -> None:
        good = self._get("/api/nb/crop/example?n=0").get_json()["inputs"]
        missing = {k: v for k, v in good.items() if k != "ph"}
        cases = [
            {"inputs": missing},
            {"inputs": {**good, "ph": 20}},
            {"inputs": {**good, "humidity": -5}},
            {"inputs": {**good, "N": "high"}},
            {"inputs": {**good, "K": None}},
            {"inputs": {**good, "extra": 1}},
            {"inputs": {**good, "N": True}},
            {},
        ]
        for payload in cases:
            response = self._post("/api/nb/crop/predict", payload)
            self.assertEqual(response.status_code, 400, payload)

    def test_non_json_request_is_rejected(self) -> None:
        response = self.client.post("/api/nb/crop/predict", data="x=1")
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
