# Use an ijk model with Python and ONNX

This is a small, beginner-friendly example for **classification** and **segmentation** models exported from [ijk.ai](https://ijk.ai). You do not need an ijk account or an internet connection to run predictions after the one-time package installation. No example sends your images to ijk or another service.

The repository contains only example code and guides. Your downloaded export supplies `model.onnx` and `manifest.json`; no user models are hosted here. The MIT license applies to this example code, not to the model file or third-party dependencies.

## Start here

If Python is not already installed, follow the guide for your computer: [Windows](docs/windows.md), [Mac](docs/mac.md), or [Linux](docs/linux.md). Each guide includes a route that does not require administrator access.

1. Download an **ONNX ZIP** for a saved model version from ijk and extract it. Keep `model.onnx` and `manifest.json` together. The ZIP also includes these example scripts.
2. Open a terminal in that extracted folder and create a local Python environment. For Windows Command Prompt:

   ```bat
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   .venv\Scripts\python.exe predict.py . path\to\photo.jpg
   ```

   For Mac or Linux:

   ```sh
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python predict.py . path/to/photo.jpg
   ```

3. Open the annotated center-crop image in `results/`. For segmentation, there is also a raw `.npy` mask. `-1` means background; `0`, `1`, and so on are indices in the export's `manifest.json` labels.

To try a whole folder, use `predict.py . path/to/images`. Files in subfolders are included and annotated results keep that folder structure. Existing results are never overwritten; repeated runs add `_2`, `_3`, and so on to the filename. To use a different export, pass its folder or `manifest.json` as the first argument.

## Live webcam

After installing requirements, run:

```sh
.venv/bin/python webcam.py . --every 5
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. Press **Q** or **Escape**, or close the window, to quit. The left panel is the live center crop. The right panel holds the most recently analyzed frame, so a segmentation overlay is never drawn onto a different, newer frame. Every fifth frame is eligible for prediction; if the CPU is still busy, that frame is skipped instead of queuing work or freezing the camera. Change `--every 5` to `--every 10` if your computer needs a slower rate. Try `--camera 1` if the default camera is not the right one.

## What the results mean

- A classification score is the model's output among its saved labels, **not measured accuracy**.
- A segmentation overlay colors the square center crop sent to the model. It does not claim to analyze the whole original photo. The `.npy` mask is at the model's native decoder resolution; the overlay enlarges it with nearest-neighbor sampling.
- The downloaded manifest controls the resize, crop, normalization, label order, and saved threshold. The scripts verify the model file's SHA-256 checksum before use.
- A saved model version and its export are fixed. If you change the model on ijk, download a new export for that version.

For problems with setup, camera permissions, missing packages, or restricted college computers, see the platform guide above. ONNX is a portable model format; these examples use CPU inference on Windows, Mac, or Linux.
