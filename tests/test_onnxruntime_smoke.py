"""Exercise the learner scripts with a tiny real ONNX model, built at test time."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

import demo
import predict


def make_tiny_export(folder):
    """Create a two-class graph with no downloaded weights or binary fixture."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
    probabilities = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, [1, 2]
    )
    indices = helper.make_tensor("indices", TensorProto.INT64, [2], [0, 2])
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["means"], axes=[2, 3], keepdims=0),
            helper.make_node("Gather", ["means", "indices"], ["selected"], axis=1),
            helper.make_node("Softmax", ["selected"], ["probabilities"], axis=1),
        ],
        "tiny_color_classifier",
        [images],
        [probabilities],
        [indices],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    # Older ONNX Runtime builds may not yet support the newest ONNX IR version.
    model.ir_version = 9
    onnx.checker.check_model(model)
    model_path = folder / "model.onnx"
    onnx.save(model, model_path)
    manifest = {
        "task": {"mode": "classification"},
        "labels": ["red", "blue"],
        "preprocessing": {
            "timm_config": {
                "input_size": [3, 4, 4],
                "crop_pct": 1.0,
                "crop_mode": "center",
                "interpolation": "bicubic",
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            }
        },
        "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def make_tiny_segmentation_export(folder):
    """Create a small graph with the two outputs expected by segmentation."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
    mask = helper.make_tensor_value_info("mask", TensorProto.INT64, [1, 2, 2])
    confidence = helper.make_tensor_value_info(
        "class_confidence", TensorProto.FLOAT, [1, 2]
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["mask"],
                value=helper.make_tensor("mask_pixels", TensorProto.INT64, [1, 2, 2], [0, 1, -1, 1]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["class_confidence"],
                value=helper.make_tensor("confidence_values", TensorProto.FLOAT, [1, 2], [0.75, 0.9]),
            ),
        ],
        "tiny_segmenter",
        [images],
        [mask, confidence],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    model_path = folder / "model.onnx"
    onnx.save(model, model_path)
    manifest = {
        "task": {"mode": "segmentation"},
        "labels": ["red", "blue"],
        "preprocessing": {
            "timm_config": {
                "input_size": [3, 4, 4],
                "crop_pct": 1.0,
                "crop_mode": "center",
                "interpolation": "bicubic",
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            }
        },
        "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class OnnxRuntimeSmokeTests(unittest.TestCase):
    def test_real_runtime_and_image_script(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            export = folder / "export"
            export.mkdir()
            make_tiny_export(export)
            image_path = folder / "sample.png"
            Image.new("RGB", (6, 4), "red").save(image_path)

            model = demo.Model(export)
            crop, prediction = model.predict(Image.open(image_path))
            self.assertEqual(crop.size, (4, 4))
            self.assertEqual(prediction["scores"].shape, (2,))
            self.assertEqual(int(np.argmax(prediction["scores"])), 0)
            self.assertAlmostEqual(float(np.sum(prediction["scores"])), 1.0, places=5)

            results = folder / "results"
            predict.main([str(export), str(image_path), "--output", str(results)])
            annotated = results / "sample.png.png"
            self.assertTrue(annotated.is_file())
            with Image.open(annotated) as result:
                self.assertEqual(result.size, (4, 4))

    def test_real_segmentation_runtime_and_image_script(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            export = folder / "export"
            export.mkdir()
            make_tiny_segmentation_export(export)
            image_path = folder / "sample.png"
            Image.new("RGB", (6, 4), "white").save(image_path)

            model = demo.Model(export)
            with Image.open(image_path) as image:
                crop, prediction = model.predict(image)
            self.assertEqual(crop.size, (4, 4))
            np.testing.assert_array_equal(prediction["mask"], [[0, 1], [-1, 1]])
            np.testing.assert_allclose(prediction["scores"], [0.75, 0.9])

            results = folder / "results"
            predict.main([str(export), str(image_path), "--output", str(results)])
            self.assertTrue((results / "sample.png.png").is_file())
            np.testing.assert_array_equal(
                np.load(results / "sample.png.npy", allow_pickle=False),
                [[0, 1], [-1, 1]],
            )


if __name__ == "__main__":
    unittest.main()
