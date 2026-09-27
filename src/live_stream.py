from flask import Flask, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

WIDTH = 640
HEIGHT = 360


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.resize(
            frame,
            (WIDTH, HEIGHT)
        )

        # JPEG encode
        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():

    return """
    <html>
        <body>
            <h1>EPV Live Stream</h1>

            <img
                src="/video"
                width="640"
                height="360"
            >
        </body>
    </html>
    """


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    print()
    print("===================================")
    print("EPV STREAM SERVER")
    print("===================================")
    print()
    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )