import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # use 'yolov8s.pt' for more accuracy

# Open video file or webcam
cap = cv2.VideoCapture("istockphoto-2213958721-640_adpp_is.mp4")  # or use 0 for webcam

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize for performance
MAX_WIDTH = 1280
MAX_HEIGHT = 720
scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
new_size = (int(width * scale), int(height * scale))

# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
out = cv2.VideoWriter("output_with_boxes.mp4", fourcc, fps, new_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, new_size)

    # YOLOv8 detection
    results = model(resized_frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf >= 0.45:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show and save
    cv2.imshow("resQeye Detection", resized_frame)
    out.write(resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
