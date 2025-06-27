import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt for more accuracy but slower

# Webcam or video file
cap = cv2.VideoCapture("istockphoto-2213958721-640_adpp_is.mp4")  # you can use 0 for webcam

# naklebi resolution = meti performance
MAX_WIDTH = 1280
MAX_HEIGHT = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame before detection
    height, width = frame.shape[:2]
    scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
    new_size = (int(width * scale), int(height * scale))
    resized_frame = cv2.resize(frame, new_size)

    # Run YOLOv8 detection
    results = model(resized_frame)[0]  # result for this frame

    # Process detections
    for box in results.boxes:
        cls = int(box.cls[0])  # class ID
        conf = float(box.conf[0])  # confidence
        if cls == 0 and conf >= 0.3:  # 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show result
    cv2.imshow("resQeye - Person Detection (YOLOv8)", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
