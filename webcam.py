"""Show the live camera and the last frame analyzed by an ijk ONNX export."""

import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from demo import Model, annotate, prepare_image
from PIL import Image

WINDOW = "ijk | live camera / last analyzed frame"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Try an ijk ONNX export on your webcam.")
    parser.add_argument("export", help="Export folder or manifest.json")
    parser.add_argument("--every", type=int, default=5, help="Predict every Nth frame (default: 5)")
    parser.add_argument("--camera", type=int, default=0, help="Camera number (default: 0)")
    args = parser.parse_args(argv)
    if args.every < 1:
        parser.error("--every must be at least 1")
    model = Model(args.export)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit("Could not open camera. Check permission or try --camera 1.")
    print("Press Q or Escape in the camera window to quit. Frames stay on this computer.")
    count = 0
    last_prediction = None
    last_frame = None
    try:
        # One worker keeps slow CPU inference from freezing the live camera panel.
        with ThreadPoolExecutor(max_workers=1) as worker:
            pending = None
            pending_frame = None
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera stopped sending frames.")
                    break
                count += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                # Both panels show the square crop, not the uncropped camera scene.
                _, crop = prepare_image(image, model.config)
                live = np.asarray(crop).copy()
                cv2.putText(
                    live,
                    "Live center crop",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                if pending is not None and pending.done():
                    analyzed_crop, prediction = pending.result()
                    last_prediction = np.asarray(annotate(analyzed_crop, prediction, model.labels))
                    last_frame = pending_frame
                    pending = None
                if (count - 1) % args.every == 0 and pending is None:
                    pending_frame = count
                    pending = worker.submit(model.predict, image.copy())
                shown = last_prediction.copy() if last_prediction is not None else live.copy()
                label = f"Analyzed frame {last_frame}" if last_frame else "Analyzing..."
                cv2.putText(
                    shown,
                    label,
                    (8, shown.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                paired = np.concatenate((live, shown), axis=1)
                cv2.imshow(WINDOW, cv2.cvtColor(paired, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Some backends raise after the user closes the window.
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
