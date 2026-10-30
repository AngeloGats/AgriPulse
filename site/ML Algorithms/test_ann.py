"""Focused tests for the leaf-image Neural Network demo routes."""

from __future__ import annotations

import io
import unittest

import pandas as pd
from PIL import Image

from ann import DATA_DIR, MODEL_DIR
from app import app


def image_bytes(size=(100, 100), fmt="PNG", color=(60, 140, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, fmt)
    return buffer.getvalue()


class AnnDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def _get(self, path: str):
        response = self.client.get(path)
        self.addCleanup(response.close)
        return response

    def _sample(self, path):
        response = self.client.post("/api/ann/predict", json={"sample": path})
        self.addCleanup(response.close)
        return response

    def _upload(self, data: bytes, name="leaf.png", mime="image/png"):
        response = self.client.post(
            "/api/ann/predict", data={"image": (io.BytesIO(data), name, mime)}, content_type="multipart/form-data"
        )
        self.addCleanup(response.close)
        return response

    def test_page_and_assets_load(self) -> None:
        page = self._get("/neural-network")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Neural Network", page.data)
        self.assertIn(b"How does a neural network recognize a leaf?", page.data)
        self.assertEqual(page.data.count(b'class="ann-thumb"'), 20)
        for asset in ("ann.css", "ann.js"):
            self.assertEqual(self._get(f"/ml-static/{asset}").status_code, 200, asset)

    def test_training_endpoint(self) -> None:
        data = self._get("/api/ann/training").get_json()
        self.assertGreaterEqual(len(data["history"]), 10)
        self.assertEqual(len(data["classes"]), 5)
        self.assertEqual(sum(map(sum, data["confusion_matrix"])), 900)

    def test_samples_match_predictions_saved_by_training(self) -> None:
        saved = pd.read_csv(MODEL_DIR / "leaf_images_test_predictions.csv")
        self.assertEqual(len(saved), 900)
        for _, row in saved.iloc[::15].iterrows():  # every 15th photo keeps the run short
            data = self._sample(row["path"]).get_json()
            self.assertEqual(data["prediction"], row["predicted"], row["path"])
            self.assertEqual(data["actual"], row["actual"])

    def test_scores_and_heatmap_are_valid(self) -> None:
        data = self._sample(pd.read_csv(DATA_DIR / "test.csv")["path"].iloc[0]).get_json()
        self.assertAlmostEqual(sum(s["probability"] for s in data["scores"]), 1.0, places=6)
        self.assertEqual(data["scores"][0]["key"], data["prediction"])
        self.assertEqual(data["confidence"], data["scores"][0]["probability"])
        self.assertEqual(len(data["heatmap"]), 8)
        for row in data["heatmap"]:
            self.assertEqual(len(row), 8)
            self.assertTrue(all(0 <= v <= 1 for v in row))
        self.assertTrue(data["network_view"].startswith("data:image/png;base64,"))

    def test_upload_matches_sample_and_accepts_other_shapes(self) -> None:
        path = pd.read_csv(DATA_DIR / "test.csv")["path"].iloc[5]
        sample = self._sample(path).get_json()
        upload = self._upload((DATA_DIR / path).read_bytes(), "leaf.jpg", "image/jpeg").get_json()
        self.assertEqual(upload["prediction"], sample["prediction"])
        self.assertAlmostEqual(upload["confidence"], sample["confidence"], places=6)
        self.assertNotIn("actual", upload)
        for size, fmt in (((300, 120), "PNG"), ((50, 400), "JPEG"), ((1, 1), "PNG")):
            response = self._upload(image_bytes(size, fmt))
            self.assertEqual(response.status_code, 200, size)

    def test_random_sample_and_sample_image(self) -> None:
        path = self._get("/api/ann/random-sample?n=7").get_json()["path"]
        image = self._get(f"/api/ann/sample-image/{path}")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/jpeg")
        for bad in ("../../app.py", "healthy/not_there.jpg", "..%5C..%5Capp.py"):
            self.assertEqual(self._get(f"/api/ann/sample-image/{bad}").status_code, 404, bad)

    def test_invalid_requests_are_rejected(self) -> None:
        train_path = pd.read_csv(DATA_DIR / "train.csv")["path"].iloc[0]
        self.assertEqual(self._sample(train_path).status_code, 400)  # training photo, not an offered sample
        self.assertEqual(self._sample("../../app.py").status_code, 400)
        self.assertEqual(self._sample(12).status_code, 400)
        for payload in ({"sample": "x", "extra": 1}, {"nothing": 1}, [1, 2]):
            response = self.client.post("/api/ann/predict", json=payload)
            self.addCleanup(response.close)
            self.assertEqual(response.status_code, 400, payload)
        self.assertEqual(self._upload(b"not an image at all").status_code, 400)
        self.assertEqual(self._upload(image_bytes(fmt="GIF"), "a.gif", "image/gif").status_code, 400)
        no_file = self.client.post("/api/ann/predict", data={"other": "x"}, content_type="multipart/form-data")
        self.addCleanup(no_file.close)
        self.assertEqual(no_file.status_code, 400)
        not_json = self.client.post("/api/ann/predict", data="x", content_type="text/plain")
        self.addCleanup(not_json.close)
        self.assertEqual(not_json.status_code, 400)

    def test_oversized_upload_is_rejected(self) -> None:
        response = self._upload(b"\x00" * (7 * 1024 * 1024))
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
