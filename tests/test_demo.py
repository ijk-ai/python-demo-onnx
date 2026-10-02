"""Fast tests with synthetic exports; no model download or webcam is needed."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ijk_onnx_demo", ROOT / "demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)
with patch.dict(sys.modules, {"demo": demo}):
    PREDICT_SPEC = importlib.util.spec_from_file_location("ijk_onnx_predict", ROOT / "predict.py")
    predict = importlib.util.module_from_spec(PREDICT_SPEC)
    PREDICT_SPEC.loader.exec_module(predict)


CONFIG = {
    "input_size": [3, 4, 4],
    "crop_pct": 1.0,
    "crop_mode": "center",
    "interpolation": "bicubic",
    "mean": [0, 0, 0],
    "std": [1, 1, 1],
}


def make_export(folder, mode):
    model = folder / "model.onnx"
    model.write_bytes(b"synthetic test artifact")
    manifest = {
        "task": {"mode": mode},
        "labels": ["red", "blue"],
        "preprocessing": {"timm_config": CONFIG},
        "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.last_input = None
        self.options = kwargs["sess_options"]

    def run(self, names, inputs):
        self.last_input = inputs["images"]
        if names == ["probabilities"]:
            return [np.array([[0.8, 0.2]], dtype=np.float32)]
        return [
            np.array([[[-1, 0], [1, -1]]], dtype=np.int64),
            np.array([[0.7, 0.9]], dtype=np.float32),
        ]


def fake_runtime():
    return types.SimpleNamespace(
        InferenceSession=FakeSession,
        SessionOptions=types.SimpleNamespace,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_DISABLE_ALL="disabled"),
    )


class DemoTests(unittest.TestCase):
    def test_center_crop_and_normalization(self):
        pixels = np.zeros((4, 6, 3), dtype=np.uint8)
        pixels[:, 1:5, 0] = 255
        tensor, crop = demo.prepare_image(Image.fromarray(pixels), CONFIG)
        self.assertEqual(crop.size, (4, 4))
        self.assertEqual(tensor.shape, (1, 3, 4, 4))
        np.testing.assert_allclose(tensor[0, 0], 1.0)

    def test_classification_export_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_export(folder, "classification")
            fake_ort = fake_runtime()
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                model = demo.Model(folder / "manifest.json")
            self.assertEqual(model.session.options.graph_optimization_level, "disabled")
            crop, prediction = model.predict(Image.new("RGB", (6, 4), "red"))
            self.assertEqual(crop.size, (4, 4))
            self.assertEqual(demo.describe(prediction, model.labels), "red: 80.0%, blue: 20.0%")
            self.assertEqual(demo.annotate(crop, prediction, model.labels).size, (4, 4))
            (folder / "model.onnx").write_bytes(b"changed")
            with (
                patch.dict(sys.modules, {"onnxruntime": fake_ort}),
                self.assertRaisesRegex(ValueError, "SHA-256"),
            ):
                demo.Model(folder)

    def test_segmentation_ids_and_exact_crop_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            make_export(folder, "segmentation")
            with patch.dict(sys.modules, {"onnxruntime": fake_runtime()}):
                model = demo.Model(folder)
            crop, prediction = model.predict(Image.new("RGB", (6, 4), "white"))
            np.testing.assert_array_equal(prediction["mask"], [[-1, 0], [1, -1]])
            self.assertEqual(demo.annotate(crop, prediction, model.labels).size, (4, 4))

    def test_result_folder_is_not_reprocessed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "photo.jpg").write_bytes(b"not decoded in this test")
            results = folder / "results"
            results.mkdir()
            (results / "old.png").write_bytes(b"old")
            self.assertEqual(demo.image_files(folder, results), [folder / "photo.jpg"])
            with self.assertRaisesRegex(ValueError, "outside the input folder"):
                demo.image_files(folder, folder)
            with self.assertRaisesRegex(ValueError, "outside the input folder"):
                demo.image_files(results, folder)

    def test_wrong_or_broken_manifest_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text("not JSON", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                demo.Model(path)
            path.write_text('{"tflite_sha256": "test"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "TFLite export"):
                demo.Model(path)
            path.write_text('{"onnx_sha256": "test", "task": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing its task or labels"):
                demo.Model(path)

    def test_predict_preserves_old_result_and_fails_when_all_images_fail(self):
        class WorkingModel:
            mode = "classification"
            labels = ("red", "blue")

            def __init__(self, path):
                pass

            def predict(self, image):
                return image.copy(), {"mode": self.mode, "scores": np.array([0.8, 0.2])}

        class FailingModel(WorkingModel):
            def predict(self, image):
                raise ValueError("fixture failure")

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "photo.png"
            Image.new("RGB", (32, 32), "red").save(image)
            output = folder / "results"
            output.mkdir()
            old = output / "photo.png.png"
            old.write_bytes(b"original")
            with patch.object(predict, "Model", WorkingModel):
                predict.main(["unused-export", str(image), "--output", str(output)])
            self.assertEqual(old.read_bytes(), b"original")
            self.assertTrue((output / "photo.png_2.png").is_file())
            with (
                patch.object(predict, "Model", FailingModel),
                self.assertRaisesRegex(SystemExit, "No images were processed"),
            ):
                predict.main(["unused-export", str(image), "--output", str(output)])

    def test_webcam_keeps_reading_while_predicting_and_stops_on_window_close(self):
        second_frame = threading.Event()
        prediction_timed_out = []

        class Camera:
            count = 0
            released = False

            def isOpened(self):
                return True

            def read(self):
                self.count += 1
                if self.count >= 2:
                    second_frame.set()
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        class SlowModel:
            config = CONFIG
            labels = ("red", "blue")

            def __init__(self, path):
                pass

            def predict(self, image):
                if not second_frame.wait(timeout=1):
                    prediction_timed_out.append(True)
                crop = demo.prepare_image(image, self.config)[1]
                return crop, {"mode": "classification", "scores": np.array([0.8, 0.2])}

        camera = Camera()
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda index: camera,
            COLOR_BGR2RGB=1,
            COLOR_RGB2BGR=2,
            FONT_HERSHEY_SIMPLEX=0,
            WND_PROP_VISIBLE=0,
            error=RuntimeError,
            cvtColor=lambda frame, code: frame,
            putText=lambda *args: None,
            imshow=lambda *args: None,
            waitKey=lambda *args: 0,
            getWindowProperty=lambda *args: 0 if camera.count >= 3 else 1,
            destroyAllWindows=lambda: None,
        )
        with patch.dict(sys.modules, {"demo": demo, "cv2": fake_cv2}):
            webcam_spec = importlib.util.spec_from_file_location(
                "ijk_onnx_webcam", ROOT / "webcam.py"
            )
            webcam = importlib.util.module_from_spec(webcam_spec)
            webcam_spec.loader.exec_module(webcam)
        with patch.object(webcam, "Model", SlowModel):
            webcam.main(["unused-export", "--every", "1"])
        self.assertGreaterEqual(camera.count, 3)
        self.assertTrue(camera.released)
        self.assertFalse(prediction_timed_out)

    def test_webcam_default_samples_the_fifth_frame(self):
        analyzed = []

        class Camera:
            count = 0

            def isOpened(self):
                return True

            def read(self):
                self.count += 1
                return True, np.full((8, 8, 3), self.count, dtype=np.uint8)

            def release(self):
                pass

        class Model:
            config = CONFIG
            labels = ("red", "blue")

            def __init__(self, path):
                pass

            def predict(self, image):
                analyzed.append(int(np.asarray(image)[0, 0, 0]))
                crop = demo.prepare_image(image, self.config)[1]
                return crop, {"mode": "classification", "scores": np.array([0.8, 0.2])}

        camera = Camera()
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda index: camera,
            COLOR_BGR2RGB=1,
            COLOR_RGB2BGR=2,
            FONT_HERSHEY_SIMPLEX=0,
            WND_PROP_VISIBLE=0,
            error=RuntimeError,
            cvtColor=lambda frame, code: frame,
            putText=lambda *args: None,
            imshow=lambda *args: None,
            waitKey=lambda *args: 0,
            getWindowProperty=lambda *args: 0 if camera.count >= 6 else 1,
            destroyAllWindows=lambda: None,
        )
        with patch.dict(sys.modules, {"demo": demo, "cv2": fake_cv2}):
            webcam_spec = importlib.util.spec_from_file_location(
                "ijk_onnx_webcam_fifth", ROOT / "webcam.py"
            )
            webcam = importlib.util.module_from_spec(webcam_spec)
            webcam_spec.loader.exec_module(webcam)
        with patch.object(webcam, "Model", Model):
            webcam.main(["unused-export"])
        self.assertEqual(analyzed, [5])


if __name__ == "__main__":
    unittest.main()
