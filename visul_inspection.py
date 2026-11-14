import cv2

#load camera
cap = cv2.VideoCapture(0)

# check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break



    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()