import cv2
from ultralytics import YOLO
from collections import deque, Counter

# Load trained model
model = YOLO("RealtimeSign/best (1).pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

print("Press 'q' to quit")

#Temporal smoothing parameters
PREDICTION_WINDOW = 7
CONF_THRESHOLD = 0.6
prediction_history = deque(maxlen=PREDICTION_WINDOW)

def temporal_smoothing(new_label):
    prediction_history.append(new_label)
    return Counter(prediction_history).most_common(1)[0][0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            raw_label = model.names[cls]
            confidence = float(box.conf[0])

            if confidence >= CONF_THRESHOLD:
                label = temporal_smoothing(raw_label)
            else:
                label = raw_label

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Sign Language Detection (YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
