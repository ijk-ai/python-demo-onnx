# Mac: from download to first prediction

You need Python 3.12, internet once for packages, and permission to run programs in your account. Inference runs on your Mac CPU. The example does not upload your photos.

## 1. Check Python

Open **Terminal** (Applications → Utilities → Terminal) and run `python3.12 --version`. If it says Python 3.12, continue. Do not replace macOS's system Python.

If Python 3.12 is missing and you do not have an administrator account, use the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to install `uv` in your own account, then open a new Terminal and run:

```sh
uv python install 3.12
```

`uv` places its managed Python under your user account. If your Mac blocks the installer or network downloads, ask your device administrator; do not bypass school policy.

## 2. Extract the export and open its folder

Double-click the downloaded ijk ONNX ZIP. In Terminal, type `cd ` (including the space), drag the extracted folder from Finder into Terminal, then press Return. `ls` should list `model.onnx`, `manifest.json`, `predict.py`, and `requirements.txt`.

## 3. Install packages locally

If you already had Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If you installed Python through `uv` and `python3.12` is not on your command path:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The `.venv` folder belongs to this download. Packages are installed once; inference then works offline.

## 4. Run it

Put a photo in the extracted folder, then run:

```sh
.venv/bin/python predict.py . photo.jpg
.venv/bin/python webcam.py . --every 5
```

Open `results/photo.jpg.png`. The image shows only the square center crop sent to the model. For segmentation, `results/photo.jpg.npy` stores raw IDs (`-1` for background, `0` for the first manifest label). Press **Q** or **Escape** in the webcam window to quit.

## If something goes wrong

- **Camera is black or unavailable**: allow Terminal or your terminal app to use the camera in System Settings → Privacy & Security → Camera; close other camera apps. Try `--camera 1` for a second camera.
- **“No module named ...”**: repeat the install command with the matching `.venv` Python.
- **Permission or security warning**: use a folder in your account and follow your organization's policy. Do not disable macOS security protections to run this example.
- **Slow webcam**: use `--every 10`.

See the [main README](../README.md) for score and mask explanations.
