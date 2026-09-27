import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading

from stream_server import start_server, update_frame


MODEL_PATH = "models/yolo26s-seg.pt"

WIDTH = 640
HEIGHT = 360

BACKGROUND_FRAMES = 60


def get_person_mask(result, width, height):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if result.masks is None:
        return mask

    masks = result.masks.data.cpu().numpy()

    for person_mask in masks:

        person_mask = cv2.resize(
            person_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        mask[person_mask > 0.5] = 255

    # -----------------------------
    # CLEAN MASK
    # -----------------------------

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    # Fill small holes
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Remove small isolated regions
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Expand mask slightly
    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # Smooth edges
    mask = cv2.GaussianBlur(
        mask,
        (9, 9),
        0
    )

    return mask


def main():
    
    server_thread = threading.Thread(
        target=start_server,
        daemon=True
    )

    server_thread.start()

    print("Loading YOLO...")

    model = YOLO(MODEL_PATH)

    # -----------------------------
    # GPU
    # -----------------------------

    if torch.cuda.is_available():

        device = 0

        print(
            "Using GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = "cpu"

        print("WARNING: CUDA is not available.")
        print("Using CPU.")

    # -----------------------------
    # CAMERA
    # -----------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    # -----------------------------
    # BACKGROUND CALIBRATION
    # -----------------------------

    print()
    print("===================================")
    print("BACKGROUND CALIBRATION")
    print("===================================")
    print()
    print("STEP AWAY FROM THE CAMERA.")
    print("Make sure NO PERSON is visible.")
    print()

    background_frames = []

    while len(background_frames) < BACKGROUND_FRAMES:

        success, frame = cap.read()

        if not success:
            continue

        # Force exact resolution
        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT)
        )

        results = model.predict(
            frame,
            classes=[0],
            conf=0.25,
            device=device,
            verbose=False
        )

        mask = get_person_mask(
            results[0],
            WIDTH,
            HEIGHT
        )

        person_pixels = cv2.countNonZero(
            mask
        )

        total_pixels = WIDTH * HEIGHT

        person_ratio = (
            person_pixels / total_pixels
        )

        # Accept only clean frames
        if person_ratio < 0.002:

            background_frames.append(
                frame.copy()
            )

            print(
                f"Clean background frame "
                f"{len(background_frames)}/"
                f"{BACKGROUND_FRAMES}"
            )

        else:

            print(
                "Person detected - frame rejected"
            )

        cv2.imshow(
            "EPV - Background Calibration",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("0"):

            cap.release()
            cv2.destroyAllWindows()

            return

    # -----------------------------
    # CREATE BACKGROUND
    # -----------------------------

    print()
    print("Building background...")

    background = np.median(
        np.stack(background_frames),
        axis=0
    ).astype(np.uint8)

    print("Background created.")

    cv2.imwrite(
        "outputs/background.jpg",
        background
    )

    print("Saved background to:")
    print("outputs/background.jpg")

    cv2.destroyWindow(
        "EPV - Background Calibration"
    )

    # -----------------------------
    # LIVE REMOVAL
    # -----------------------------

    print()
    print("===================================")
    print("LIVE PEOPLE REMOVAL")
    print("===================================")
    print("Walk into the camera.")
    print("Press 0 to quit.")
    print()

    previous_time = time.time()

    # Previous mask for temporal smoothing
    previous_mask = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.float32
    )

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Force exact resolution
        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT)
        )

        # -----------------------------
        # YOLO TRACKING
        # -----------------------------

        results = model.predict(
        frame,
        classes=[0],
        conf=0.30,
        device=device,
        verbose=False
        )

        # -----------------------------
        # PERSON MASK
        # -----------------------------

        mask = get_person_mask(
            results[0],
            WIDTH,
            HEIGHT
        )

        current_mask = mask.astype(
            np.float32
        )

        # -----------------------------
        # TEMPORAL SMOOTHING
        # -----------------------------

        previous_mask = (
            0.35 * previous_mask
            +
            0.65 * current_mask
        )

        smooth_mask = previous_mask

        # -----------------------------
        # CONVERT MASK TO ALPHA
        # -----------------------------

        alpha = smooth_mask / 255.0

        alpha = alpha[:, :, None]

        # -----------------------------
        # BACKGROUND REPLACEMENT
        # -----------------------------

        output = (
            frame.astype(np.float32)
            * (1 - alpha)
            +
            background.astype(np.float32)
            * alpha
        )

        output = np.clip(
            output,
            0,
            255
        ).astype(np.uint8)

        # -----------------------------
        # FPS
        # -----------------------------

        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            0.0001
        )

        previous_time = current_time

        # -----------------------------
        # DISPLAY MASK
        # -----------------------------
        
        update_frame(output)

        # -----------------------------
        # DISPLAY FINAL OUTPUT
        # -----------------------------

        cv2.imshow(
            "EPV - Live People Removal",
            output
        )
        output = np.clip(output,0,255).astype(np.uint8)
        
        update_frame(output)

        # -----------------------------
        # QUIT
        # -----------------------------

        if cv2.waitKey(1) & 0xFF == ord("0"):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()