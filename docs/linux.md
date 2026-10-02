# Linux: from download to first prediction

You need a desktop Linux session, Python 3.12, internet once for packages, and a camera if you want the live example. Predictions run locally on your CPU.

## 1. Check Python

Open a terminal and run `python3.12 --version`. If Python 3.12 is installed, continue. Do not replace the system Python or use `sudo pip`.

Without administrator access, follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to install `uv` for your user, open a new terminal, then run:

```sh
uv python install 3.12
```

If the machine blocks local programs or package downloads, ask IT for an approved setup. No root access or package-manager changes are required by the example itself.

## 2. Extract the export

Extract the ijk ONNX ZIP to a folder in your home directory, open a terminal there, and check that `model.onnx`, `manifest.json`, `predict.py`, and `requirements.txt` are together. One way is to type `cd `, drag the extracted folder into the terminal, and press Enter.

## 3. Install packages locally

If you already had Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If you installed Python with `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

This creates `.venv` in the export folder and needs no system-wide install.

## 4. Run it

```sh
.venv/bin/python predict.py . photo.jpg
.venv/bin/python webcam.py . --every 5
```

The annotated square center crop goes to `results/photo.jpg.png`. A segmentation mask also goes to `results/photo.jpg.npy`; `-1` means background and other values index the manifest labels. Press **Q** or **Escape** to stop the webcam.

## If something goes wrong

- **“No module named ...”**: reinstall requirements using the same `.venv` Python.
- **Camera unavailable**: ensure another program is not using it; try `--camera 1`. You may need your administrator to allow webcam access to your account.
- **No display / OpenCV error**: `webcam.py` needs a graphical desktop session; `predict.py` works without a camera or display.
- **Slow webcam**: use `--every 10`.

See the [main README](../README.md) for result meanings.
