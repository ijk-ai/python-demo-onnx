"""Small, reusable reader for an ijk ONNX export. No ijk account is needed."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [(0, 181, 190), (255, 107, 82), (106, 83, 204), (236, 173, 47)]


def read_manifest(path):
    """Reject incomplete or wrong-format downloads with a useful explanation."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.json is not valid JSON. Re-extract the ONNX ZIP.") from error
    if not isinstance(manifest, dict):
        raise TypeError("manifest.json must contain a model description. Re-extract the ONNX ZIP.")
    if "onnx_sha256" not in manifest:
        if "tflite_sha256" in manifest:
            raise ValueError("This is a TFLite export. Download ONNX or use the TFLite examples.")
        raise ValueError("This is not a complete ijk ONNX export. Re-download the ONNX ZIP.")
    task = manifest.get("task")
    labels = manifest.get("labels")
    preprocessing = manifest.get("preprocessing")
    config = preprocessing.get("timm_config") if isinstance(preprocessing, dict) else None
    if not isinstance(task, dict) or not isinstance(labels, list) or not labels:
        raise ValueError("Export manifest is missing its task or labels. Re-download the ONNX ZIP.")
    if task.get("mode") not in {"classification", "segmentation"}:
        raise ValueError("This ONNX example supports classification and segmentation exports only.")
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError("Export manifest labels must be non-empty text.")
    if not isinstance(config, dict) or not all(
        key in config for key in ("input_size", "crop_pct", "interpolation", "mean", "std")
    ):
        raise ValueError(
            "Export manifest is missing image preprocessing. Re-download the ONNX ZIP."
        )
    size = config["input_size"]
    if not isinstance(size, list | tuple) or len(size) != 3:
        raise ValueError("Export manifest has an invalid image size. Re-download the ONNX ZIP.")
    if not isinstance(config["crop_pct"], int | float) or config["crop_pct"] <= 0:
        raise ValueError("Export manifest has an invalid crop setting. Re-download the ONNX ZIP.")
    if any(
        not isinstance(config[key], list | tuple) or len(config[key]) != 3
        for key in ("mean", "std")
    ):
        raise ValueError(
            "Export manifest has invalid image normalization. Re-download the ONNX ZIP."
        )
    return manifest


def image_files(path, output_dir):
    """Accept one image or a directory; do not reprocess an earlier results folder."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() in IMAGE_TYPES:
        return [path]
    if path.is_dir():
        output_dir = Path(output_dir).resolve()
        source_dir = path.resolve()
        if output_dir == source_dir or output_dir in source_dir.parents:
            raise ValueError(
                "Choose a results folder outside the input folder, not the same folder or its parent."
            )
        return [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file()
            and item.suffix.lower() in IMAGE_TYPES
            and output_dir not in item.resolve().parents
        ]
    raise ValueError("Choose an image (JPG/PNG/BMP/WebP) or a folder of images.")


def prepare_image(image, config):
    """Use the same RGB resize, square center crop and normalization as preview."""
    channels, height, width = config["input_size"]
    if channels != 3 or height != width or config.get("crop_mode", "center") != "center":
        raise ValueError("This export needs a square RGB center crop.")
    if config["interpolation"] != "bicubic":
        raise ValueError("This example supports bicubic interpolation only.")
    image = image.convert("RGB")
    target = int(height / config["crop_pct"])
    old_width, old_height = image.size
    if old_width <= old_height:
        new_size = (target, int(target * old_height / old_width))
    else:
        new_size = (int(target * old_width / old_height), target)
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    left = round((new_size[0] - width) / 2)
    top = round((new_size[1] - height) / 2)
    crop = resized.crop((left, top, left + width, top + height))
    pixels = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.asarray(config["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(config["std"], dtype=np.float32)[:, None, None]
    tensor = np.ascontiguousarray(((pixels - mean) / std)[None, ...])
    return tensor, crop


class Model:
    def __init__(self, export_path):
        export_path = Path(export_path)
        manifest_path = export_path / "manifest.json" if export_path.is_dir() else export_path
        if manifest_path.name != "manifest.json" or not manifest_path.is_file():
            raise ValueError("Give the exported folder or its manifest.json file.")
        self.manifest = read_manifest(manifest_path)
        self.mode = self.manifest["task"]["mode"]
        self.labels = self.manifest["labels"]
        self.config = self.manifest["preprocessing"]["timm_config"]
        model_path = manifest_path.with_name("model.onnx")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found next to manifest: {model_path}")
        with model_path.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        if digest != self.manifest["onnx_sha256"]:
            raise ValueError("model.onnx does not match the export manifest (SHA-256).")
        # No network, cloud endpoint, or GPU provider is used by these examples.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def predict(self, image):
        tensor, crop = prepare_image(image, self.config)
        if self.mode == "classification":
            (scores,) = self.session.run(["probabilities"], {"images": tensor})
            return crop, {"mode": self.mode, "scores": scores[0]}
        mask, confidence = self.session.run(["mask", "class_confidence"], {"images": tensor})
        return crop, {"mode": self.mode, "mask": mask[0], "scores": confidence[0]}


def annotate(crop, prediction, labels):
    """Show only the square crop that actually went into the model."""
    result = crop.copy()
    if prediction["mode"] == "segmentation":
        # Native decoder pixels are expanded with nearest-neighbor to preserve IDs.
        mask = Image.fromarray(prediction["mask"].astype(np.int32), mode="I")
        mask = np.asarray(mask.resize(crop.size, Image.Resampling.NEAREST))
        pixels = np.asarray(result).copy()
        for index in range(len(labels)):
            selected = mask == index
            if selected.any():
                color = np.asarray(COLORS[index % len(COLORS)])
                pixels[selected] = (pixels[selected] * 0.45 + color * 0.55).astype(np.uint8)
        result = Image.fromarray(pixels)
        title = "Segmentation: " + ", ".join(
            labels[index] for index in range(len(labels)) if (mask == index).any()
        )
        if title == "Segmentation: ":
            title = "Segmentation: background"
    else:
        top = int(np.argmax(prediction["scores"]))
        title = f"{labels[top]}  {float(prediction['scores'][top]):.0%}"
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 25), fill=(5, 28, 34))
    draw.text((7, 5), title[:60], fill="white")
    return result


def describe(prediction, labels):
    """Scores are model outputs, not measured accuracy."""
    ranking = np.argsort(-prediction["scores"])
    return ", ".join(f"{labels[i]}: {float(prediction['scores'][i]):.1%}" for i in ranking)
