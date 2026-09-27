import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)
frame_number = 0


def update_frame(frame):
    global latest_frame
    global frame_number

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 75]
    )

    if not success:
        return

    with frame_condition:
        latest_frame = buffer.tobytes()
        frame_number += 1
        frame_condition.notify_all()


def generate():
    global latest_frame
    global frame_number

    last_frame_number = -1

    while True:

        with frame_condition:

            while (
                latest_frame is None
                or frame_number == last_frame_number
            ):
                frame_condition.wait()

            frame = latest_frame
            current_frame_number = frame_number

        last_frame_number = current_frame_number

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Pragma: no-cache\r\n"
            b"\r\n"
            + frame
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>EPV Live Stream</title>
    </head>

    <body>
        <h1>EPV - Live People Removal</h1>

        <img
            src="/video"
            width="640"
            height="360"
            style="max-width:100%; height:auto;"
        >
    </body>
    </html>
    """


@app.route("/video")
def video():

    response = Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


def start_server():

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        use_reloader=False
    )