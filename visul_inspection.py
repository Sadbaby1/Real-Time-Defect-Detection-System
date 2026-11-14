import cv2
from ultralytics import YOLO

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
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera Feed", annotated_frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
