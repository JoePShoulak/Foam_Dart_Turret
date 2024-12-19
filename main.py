import cv2
import mediapipe as mp
import math
import numpy as np

SCALE_FACTOR = 4  # Define your scale factor as needed

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Check if the video capture device is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open webcam. Please check your device.")
    exit()

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to display stats and a face sketch
def display_stats_and_sketch(ipd):
    # Create a resized canvas for stats and sketch (adjusted dimensions)
    stats_window = np.ones((600, 1200, 3), dtype=np.uint8) * 255

    # Draw an ellipse to represent the head in the left column
    head_center = (300, 300)  # Center of the "head" in the left column
    head_axes = (120, 180)  # Width and height of the ellipse
    cv2.ellipse(stats_window, head_center, head_axes, 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)

    # Adjust eyeline position to represent eye level
    eyeline_y = head_center[1] - 90  # Move upward closer to the top of the ellipse
    eye_x_offset = int(ipd / 2)  # Half the IPD determines the horizontal offset
    left_eye_sketch = (head_center[0] - eye_x_offset, eyeline_y)
    right_eye_sketch = (head_center[0] + eye_x_offset, eyeline_y)

    # Draw eyes and line between them
    cv2.circle(stats_window, left_eye_sketch, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(stats_window, right_eye_sketch, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.line(stats_window, left_eye_sketch, right_eye_sketch, (255, 0, 0), 3, cv2.LINE_AA)

    # Display stats text in the right column
    text = [f"Interpupillary Distance (IPD): {ipd:.2f} px"]

    # Position the stats slightly left and vertically centered in the right column
    text_start_x = 700  # Start closer to the center
    text_start_y = 300 - (len(text) * 30 // 2)  # Center vertically
    text_line_height = 30
    for i, line in enumerate(text):
        cv2.putText(stats_window, line, (text_start_x, text_start_y + i * text_line_height), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    # Show the stats and sketch window
    cv2.imshow("Stats and Sketch", stats_window)

# Main Loop
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Convert frame to RGB (Mediapipe uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and landmarks
    results = face_mesh.process(rgb_frame)

    # Draw landmarks and calculate stats if faces are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left eye center
            left_eye = [
                face_landmarks.landmark[i]
                for i in [33, 133, 159, 145]  # IDs for key left eye points
            ]
            left_eye_center = (
                sum([point.x for point in left_eye]) / len(left_eye),
                sum([point.y for point in left_eye]) / len(left_eye),
            )

            # Get right eye center
            right_eye = [
                face_landmarks.landmark[i]
                for i in [362, 263, 386, 374]  # IDs for key right eye points
            ]
            right_eye_center = (
                sum([point.x for point in right_eye]) / len(right_eye),
                sum([point.y for point in right_eye]) / len(right_eye),
            )

            # Convert normalized coordinates to pixel coordinates
            left_eye_pixel = (
                int(left_eye_center[0] * frame.shape[1]),
                int(left_eye_center[1] * frame.shape[0]),
            )
            right_eye_pixel = (
                int(right_eye_center[0] * frame.shape[1]),
                int(right_eye_center[1] * frame.shape[0]),
            )

            # Calculate the Euclidean distance between the eyes
            ipd = math.sqrt(
                (right_eye_pixel[0] - left_eye_pixel[0]) ** 2
                + (right_eye_pixel[1] - left_eye_pixel[1]) ** 2
            )

            # Draw the line between eyes on the video frame
            cv2.line(frame, left_eye_pixel, right_eye_pixel, (255, 0, 0), 2)

            # Display stats and a fixed sketch
            display_stats_and_sketch(ipd)

    # Display the video feed with landmarks
    cv2.imshow("Face Detection with Landmarks", frame)

    # Ensure proper event loop handling
    key = cv2.waitKey(1)  # This processes OpenCV window events

    # Check if either window is closed manually
    if cv2.getWindowProperty("Face Detection with Landmarks", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.getWindowProperty("Stats and Sketch", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed manually. Exiting...")
        break

# Release Mediapipe resources
face_mesh.close()
# Release handle to the webcam and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
