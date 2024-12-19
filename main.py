import cv2
from face_distance import *  # Retained for future use if needed

SCALE_FACTOR = 4  # Define your scale factor as needed

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Detect faces using Haar cascades (faster than face_recognition library)
    face_cor = face_model.detectMultiScale(frame)

    # If faces are detected
    if len(face_cor) > 0:
        for (x1, y1, w, h) in face_cor:
            x2, y2 = x1 + w, y1 + h
            # Draw a box around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Uncomment the following lines to enable IPD calculation and display
            # if you plan to bring back face distance functionality
            # face_location = (y1, x2, y2, x1)  # Convert to (top, right, bottom, left) format
            # landmarks = {}  # You can compute landmarks here if needed
            # draw_face_rectangle_and_label(frame, face_location, landmarks, SCALE_FACTOR)

    # Display the resulting image
    cv2.imshow('Face Detection', frame)

    # Check if the window is closed manually
    if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed manually. Exiting...")
        break

# Release handle to the webcam and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
