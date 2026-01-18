#loading model to test
from ultralytics import YOLO
model = YOLO("C:\\Users\\TUPAK\\OneDrive\\Desktop\\ISL_Detection\\RealtimeSign\\best (1).pt")   # change to last.pt if needed
print("Model loaded successfully")