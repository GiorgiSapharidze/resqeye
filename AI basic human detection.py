import cv2
import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # 0 = adamiani

# 0 = webcam, debugistvis 'path/video.mp4' shegidzlia gamoiyeno
cap = cv2.VideoCapture("istockphoto-2213958721-640_adpp_is.mp4")  # 0 dan 4 sheidzleba iyos sheni kamera

# naklebi resolution = meti performance
MAX_WIDTH = 1280
MAX_HEIGHT = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    height, width = frame.shape[:2]
    scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
    new_size = (int(width * scale), int(height * scale))
    resized_frame = cv2.resize(frame, new_size)

    results = model(resized_frame)

    # boxebis daxatva
    for *box, conf, cls in results.xyxy[0]:
        if conf >= 0.45: # amis datweakeba sheidzleba
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, 'Person' + str(conf), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    
    cv2.imshow("resQeye - Person Detection", resized_frame)

   
    if cv2.waitKey(1) == ord('q'): # q gamosvla
        break

cap.release()
cv2.destroyAllWindows()
