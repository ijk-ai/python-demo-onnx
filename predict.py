"""Run an exported model on one image or a folder: python predict.py EXPORT IMAGE."""

import argparse
from pathlib import Path

import numpy as np
from demo import Model, annotate, describe, image_files
from PIL import Image


def unused_target(path, with_mask):
    """Keep old results intact when the same photo is tested again."""
    candidate = path
    number = 2
    while candidate.exists() or (with_mask and candidate.with_suffix(".npy").exists()):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        number += 1
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict using an ijk ONNX export, on this computer."
    )
    parser.add_argument("export", help="Export folder or manifest.json")
    parser.add_argument("images", help="Image file or folder of images")
    parser.add_argument("--output", default="results", help="Where to save annotated images")
    args = parser.parse_args(argv)

    model = Model(args.export)
    source = Path(args.images)
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        parser.error("--output must name a folder, not a file.")
    try:
        files = image_files(source, output)
    except ValueError as error:
        parser.error(str(error))
    if not files:
        parser.error("No JPG, PNG, BMP, or WebP images found in that folder.")
    completed = 0
    for path in files:
        # Keep subfolders and the original extension in the name to avoid collisions.
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        proposed = output / relative.parent / (relative.name + ".png")
        try:
            with Image.open(path) as image:
                crop, prediction = model.predict(image)
            annotated = annotate(crop, prediction, model.labels)
            proposed.parent.mkdir(parents=True, exist_ok=True)
            target = unused_target(proposed, model.mode == "segmentation")
            with target.open("xb") as stream:
                annotated.save(stream, format="PNG")
            print(f"{path}: {describe(prediction, model.labels)}")
            print(f"  Annotated center crop: {target}")
            if model.mode == "segmentation":
                raw_mask = target.with_suffix(".npy")
                with raw_mask.open("xb") as stream:
                    np.save(stream, prediction["mask"].astype(np.int16), allow_pickle=False)
                print(f"  Raw mask: {raw_mask} (-1 background, 0..N-1 label index)")
            completed += 1
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Skipped {path}: {error}")
    if not completed:
        raise SystemExit("No images were processed successfully.")


if __name__ == "__main__":
    main()
