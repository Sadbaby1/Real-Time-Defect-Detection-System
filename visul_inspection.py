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

    # identify image segmentation
    # bboxes, class_ids, segments, scores = model.predict(frame, conf=0.5)
    # results = model("path/to/image.jpg", task="semantic")
    results = model(frame, conf=0.5)
    # plot the segmented objects using built in function
    # annotated_frame = results[0].plot()

    # segement manualy
    result = results[0]

    # loop through all detected objects
    # 1. Bounding boxes, classes, and confidence scores
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
            

    """
    # loop through all detected objects
    
    for bbox, class_id, segment, score in zip(bboxes, class_ids, segments, scores):
        # draw bounding box and label
        # show bbox
        x, y, X2, Y2 = map(int, bbox)
        cv2.rectangle(frame, (x, y), (X2, Y2), (0, 255, 0), 2)

        # cv2.putText(frame, f"Class: {class_id}, Score: {score:.2f}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    """
    # Display the resulting frame
    cv2.imshow("Camera Feed",   frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
