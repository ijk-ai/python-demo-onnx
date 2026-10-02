# Windows: from download to first prediction

You need a 64-bit Windows PC, permission to run a local Python program, and internet access once to install packages. You **do not** need an administrator account for the normal steps. Predictions run on your CPU, and the example does not upload your photos.

## 1. Get Python 3.12

If `py -3.12 --version` already prints a version, continue to step 2. Otherwise download a **Python 3.12 64-bit installer** from [python.org](https://www.python.org/downloads/windows/). Choose **Install Now** for your own account, not “Install for all users.” The [official Windows guide](https://docs.python.org/3.12/using/windows.html) says this normally does not need administrator access. If Windows asks to update the system C runtime or your college blocks installers, stop and ask IT; do not try to bypass device policy.

Open a new **Command Prompt** (search the Start menu for `cmd`) and check:

```bat
py -3.12 --version
```

If `py` is not found, try `python --version`. It must say Python 3.12. If that works, substitute `python` for `py -3.12` below.

## 2. Open the downloaded model folder

Extract your ijk ONNX ZIP in File Explorer. Do not run commands while it is still inside the ZIP. You should see `model.onnx`, `manifest.json`, `predict.py`, `webcam.py`, and `requirements.txt` together. In File Explorer, open that folder, click its address bar, type `cmd`, and press Enter. Command Prompt opens in the right folder.

## 3. Install into a folder you own

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.venv` is a private folder inside your export. Nothing is installed for all users. Installation needs internet; the later predictions do not. You do not need to activate the environment or change PowerShell settings.

## 4. Try an image, then the webcam

Put `photo.jpg` beside the scripts, then run:

```bat
.venv\Scripts\python.exe predict.py . photo.jpg
.venv\Scripts\python.exe webcam.py . --every 5
```

Open `results\photo.jpg.png` to see the annotated **center crop**. For segmentation, `results\photo.jpg.npy` is the raw mask; its `-1` pixels mean background. The webcam predicts every fifth frame by default. Press **Q** or **Escape** to close it. To run a folder of photos, replace `photo.jpg` with its folder path, in quotes if it contains spaces.

## If something goes wrong

- **“No module named ...”**: run the install command again using the same `.venv\Scripts\python.exe`; do not use a different Python.
- **“Could not open camera”**: close other camera apps, check Windows camera permissions, or add `--camera 1`. A managed college PC may block camera use; ask IT.
- **“Access denied” or installation blocked**: use a folder under your own Documents, not `Program Files`. If the school blocks Python or package downloads, ask IT for permission or use an approved PC; there is no safe bypass in this guide.
- **“Model does not match manifest”**: re-extract the complete ZIP. Keep the model file and manifest from the same export.
- **Slow webcam**: try `--every 10` and close other CPU-heavy apps.

The [main README](../README.md) explains score meanings, segmentation masks, and paths to another exported model.
