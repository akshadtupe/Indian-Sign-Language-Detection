# Indian Sign Language Detection
Real-time sign language detection using a custom-trained YOLOv8 model and a webcam.
This project develops a custom real-time Indian Sign Language (ISL) recognition system using a YOLOv8-based object detection model trained entirely on a self-collected and annotated dataset, addressing the absence of publicly available ISL datasets and pretrained models. The system performs live webcam-based gesture detection with temporal smoothing to achieve stable and low-latency inference on a local machine.

# Features
- Real-time webcam inference
- Custom YOLOv8 model
- Stable predictions using temporal smoothing
- Runs locally on CPU

## Installations
```bash
pip install ultralytics opencv-python
```

## Run the Application
```bash
python web_detection.py
```

## 🧩 How It Works
1. Webcam frames are captured using OpenCV
2. YOLOv8 performs object detection on each frame
3. Detected gestures are classified into sign labels
4. Temporal smoothing stabilizes predictions across frames
5.Results are rendered in real time




