# EPV — Live People Removal

EPV is a real-time computer-vision application that detects people from a live webcam feed and replaces detected people with a previously captured clean background.

## Project Structure

```text
EPV/
├── models/
│   ├── yolo26n/
│   │   └── yolo26n-seg.pt
│   └── yolo26s/
│       └── yolo26s-seg.pt
├── src/
│   ├── live_people_removal.py
│   └── stream_server.py
├── outputs/
│   └── background.jpg
├── .venv/
└── README.md
```

## Requirements

- Windows
- Python 3.11
- Webcam
- NVIDIA GPU with CUDA support recommended
- Wi-Fi if viewing from another device

The project has been tested with an NVIDIA RTX 3060.

## 1. Activate the environment

```powershell
cd C:\Users\teenu\Documents\EPV
.\.venv\Scripts\Activate.ps1
python --version
```

Expected: Python 3.11.x.

## 2. Install dependencies

If not already installed:

```powershell
python -m pip install --upgrade pip
pip install ultralytics opencv-python flask numpy
```

Check GPU support:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

CUDA is recommended for real-time performance.

## 3. Model selection

EPV supports two YOLO segmentation models:

```text
models/yolo26n/yolo26n-seg.pt
models/yolo26s/yolo26s-seg.pt
```

### YOLO26n — Nano

Use:

```python
MODEL_PATH = "models/yolo26n/yolo26n-seg.pt"
```

Why use it:

- Faster inference
- Lower GPU memory usage
- Better suited when real-time FPS is the priority
- Good starting point for live webcam processing

### YOLO26s — Small

Use:

```python
MODEL_PATH = "models/yolo26s/yolo26s-seg.pt"
```

Why use it:

- Larger model capacity than the nano model
- Can provide better detection/segmentation in some scenes
- Useful for multiple people, smaller people, overlapping people, and more complicated boundaries
- Costs more computation and may reduce FPS

### Switching models

Open:

```text
src/live_people_removal.py
```

Change `MODEL_PATH` to either model path, save the file, and restart EPV.

There is no universal winner: test both models on the actual camera scene and compare segmentation quality and FPS.

## 4. Run EPV

```powershell
cd C:\Users\teenu\Documents\EPV
.\.venv\Scripts\Activate.ps1
python src\live_people_removal.py
```

## 5. Background calibration

When the program starts, step away from the camera and make sure no person is visible. Keep the camera stationary and lighting reasonably stable.

The program captures clean frames and creates:

```text
outputs/background.jpg
```

This image is used to replace detected people.

## 6. Live people removal

After calibration, the pipeline is:

```text
Webcam
  ↓
YOLO segmentation
  ↓
Person mask
  ↓
Mask cleanup
  ↓
Mask smoothing
  ↓
Background replacement
  ↓
Processed live video
```

Press `Q` to stop.

## 7. View on the same computer

Open in a browser or VLC:

```text
http://127.0.0.1:5000/video
```

Web interface:

```text
http://127.0.0.1:5000
```

## 8. View on iPhone / another device

Connect the PC and phone to the same Wi-Fi network. Find the PC IPv4 address:

```powershell
ipconfig
```

Example:

```text
192.168.0.30
```

On the iPhone, open:

```text
http://192.168.0.30:5000/video
```

Replace the IP with the PC's current address.

### Windows Firewall

If another device cannot connect, run Administrator PowerShell:

```powershell
New-NetFirewallRule `
  -DisplayName "EPV Live Stream 5000" `
  -Direction Inbound `
  -Protocol TCP `
  -LocalPort 5000 `
  -Action Allow
```

Check it:

```powershell
Get-NetFirewallRule -DisplayName "EPV Live Stream 5000" |
Select-Object DisplayName, Enabled, Direction, Action
```

## 9. VLC

1. Open VLC.
2. Select **Media → Open Network Stream**.
3. Enter:

```text
http://192.168.0.30:5000/video
```

4. Click **Play**.

Use the current PC IP address.

## 10. Important background limitation

The current system uses a static background captured before live processing. It works best when the camera is stationary and the background and lighting do not change significantly.

If the camera moves, zooms, or the scene changes substantially, the captured background may no longer match the current scene.

## 11. Accuracy testing

Test both models with:

- One person
- Multiple people
- People close together
- Fast movement
- A person near the image edge
- Small/distant people

Look for missing body parts, ghosting, flickering, holes, and incomplete removal. Run the same tests with YOLO26n and YOLO26s before choosing the model for deployment.

## 12. Troubleshooting

### Camera does not open

Close applications that may be using the webcam and retry.

### CUDA is unavailable

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns `False`, PyTorch is not currently using CUDA. CPU fallback is possible but may be slower.

### iPhone cannot connect

Check that:

1. PC and iPhone are on the same Wi-Fi.
2. The PC IP is correct.
3. EPV is running.
4. Port 5000 is listening.
5. Windows Firewall allows TCP 5000.

Check the port:

```powershell
netstat -ano | findstr :5000
```

Expected:

```text
TCP    0.0.0.0:5000    0.0.0.0:0    LISTENING
```

### iPhone connects but video is frozen

Use the current `stream_server.py` implementation with frame synchronization and no-cache headers.

## 13. Current workflow

```text
Activate .venv
      ↓
Select YOLO model
      ↓
Run live_people_removal.py
      ↓
Capture clean background
      ↓
Person enters camera view
      ↓
YOLO detects person
      ↓
Generate segmentation mask
      ↓
Replace person with background
      ↓
Serve processed video through Flask
      ↓
View on PC / VLC / iPhone
```

## 14. Project goal

EPV aims to provide a real-time system that detects and removes people from a live camera feed, replaces them using a clean background, uses GPU acceleration when available, and exposes the processed stream over the local network.

## Future improvements

- Better segmentation boundaries
- Improved multi-person handling
- Automatic/dynamic background handling
- Dynamic-scene inpainting
- Lower latency
- H.264/RTMP/SRT streaming
- Virtual camera output
- FPS and GPU monitoring
