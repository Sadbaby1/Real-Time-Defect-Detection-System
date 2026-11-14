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

    # Save a clean copy of the frame before drawing annotations
    clean_frame = frame.copy()

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
    # 2. Fill detected objects with semi-transparent color and extract
    if result.masks is not None:
        overlay = clean_frame.copy()
        # Binary mask for isolating the segmented objects
        mask = np.zeros(clean_frame.shape[:2], dtype=np.uint8)

        for segment in result.masks.xy:
            if len(segment) > 0:
                pts = segment.astype(np.int32)
                # Draw contour border on the  object
                cv2.polylines(
                    frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2
                )

                # Fill polygon on the overlay layer (for transparency)
                cv2.fillPoly(overlay, [pts], color=(255, 0, 0))
                # Fill polygon on the binary mask for extraction
                cv2.fillPoly(mask, [pts], 255)

                # Blend overlay with clean_frame so extracted object has the transparent mask
                alpha = 0.5
                masked_frame = cv2.addWeighted(
                    overlay, alpha, clean_frame, 1 - alpha, 0
                )

                # Extract the object WITH the transparent mask on it
                extracted_objects = cv2.bitwise_and(
                    masked_frame, masked_frame, mask=mask
                )

                # Blend overlay with camera frame for the main feed
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("Extracted objects", extracted_objects)

    # Display the resulting frame
    cv2.imshow("Camera Feed", frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
