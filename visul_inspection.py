import cv2
from ultralytics import YOLO
import numpy as np

# load camera
cap = cv2.VideoCapture(0)

# check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# load the segmentation model
model = YOLO("yolo11n-seg.pt")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Identify image segmentation
    results = model(frame, conf=0.5)
    result = results[0]

    # 1. Draw bounding boxes, classes, and confidence scores on top
    if result.boxes is not None:
        for box, cls_id, conf in zip(
            result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    # 2. Fill detected objects with semi-transparent color
    if result.masks is not None:
        overlay = frame.copy()
        for segment in result.masks.xy:
            if len(segment) > 0:
                pts = segment.astype(np.int32)
                # Fill polygon on the overlay layer
                cv2.fillPoly(overlay, [pts], color=(255, 0, 0))

        # Blend overlay with original frame (alpha=0.5 transparency)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Display the resulting frame
    cv2.imshow("Camera Feed", frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
