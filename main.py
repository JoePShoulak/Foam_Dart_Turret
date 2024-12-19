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


# Helper Functions
def get_landmark_pixel(landmarks, frame_shape):
    """Convert normalized landmarks to pixel coordinates."""
    return [
        (int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0]))
        for landmark in landmarks
    ]


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_eye_center(landmarks, frame_shape):
    """Calculate the center of the eye from its landmarks."""
    eye_pixel = get_landmark_pixel(landmarks, frame_shape)
    center_x = sum([point[0] for point in eye_pixel]) / len(eye_pixel)
    center_y = sum([point[1] for point in eye_pixel]) / len(eye_pixel)
    return int(center_x), int(center_y)


def display_stats_and_sketch(ipd, mouth_width):
    """Display stats and draw a face sketch."""
    stats_window = np.ones((600, 1200, 3), dtype=np.uint8) * 255

    # Draw an ellipse to represent the head in the left column
    head_center = (300, 300)
    head_axes = (120, 180)
    cv2.ellipse(stats_window, head_center, head_axes, 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)

    # Eyeline and sketch
    eyeline_y = head_center[1] - 45
    eye_x_offset = int(ipd / 2)
    left_eye_sketch = (head_center[0] - eye_x_offset, eyeline_y)
    right_eye_sketch = (head_center[0] + eye_x_offset, eyeline_y)
    cv2.circle(stats_window, left_eye_sketch, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(stats_window, right_eye_sketch, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.line(stats_window, left_eye_sketch, right_eye_sketch, (255, 0, 0), 3, cv2.LINE_AA)

    # Mouth line and sketch
    mouthline_y = head_center[1] + 50
    mouth_x_offset = int(mouth_width / 2)
    left_mouth_sketch = (head_center[0] - mouth_x_offset, mouthline_y)
    right_mouth_sketch = (head_center[0] + mouth_x_offset, mouthline_y)
    cv2.circle(stats_window, left_mouth_sketch, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(stats_window, right_mouth_sketch, 8, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.line(stats_window, left_mouth_sketch, right_mouth_sketch, (255, 0, 0), 3, cv2.LINE_AA)

    # Display stats text
    text = [
        f"Interpupillary Distance (IPD): {ipd:.2f} px",
        f"Mouth Width: {mouth_width:.2f} px",
    ]
    text_start_x = 650
    text_start_y = 300 - (len(text) * 30 // 2)
    text_line_height = 30
    for i, line in enumerate(text):
        cv2.putText(stats_window, line, (text_start_x, text_start_y + i * text_line_height),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

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

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 133, 159, 145]]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 263, 386, 374]]

            # Get eye centers
            left_eye_pixel = get_eye_center(left_eye_landmarks, frame.shape)
            right_eye_pixel = get_eye_center(right_eye_landmarks, frame.shape)

            # Calculate IPD
            ipd = calculate_distance(left_eye_pixel, right_eye_pixel)

            # Get mouth landmarks
            left_mouth_corner = face_landmarks.landmark[61]
            right_mouth_corner = face_landmarks.landmark[291]
            left_mouth_pixel, right_mouth_pixel = get_landmark_pixel([left_mouth_corner, right_mouth_corner], frame.shape)

            # Calculate mouth width
            mouth_width = calculate_distance(left_mouth_pixel, right_mouth_pixel)

            # Draw lines on video
            cv2.line(frame, left_eye_pixel, right_eye_pixel, (255, 0, 0), 2)
            cv2.line(frame, left_mouth_pixel, right_mouth_pixel, (255, 0, 0), 2)

            # Display stats and sketch
            display_stats_and_sketch(ipd, mouth_width)

    cv2.imshow("Face Detection with Landmarks", frame)

    key = cv2.waitKey(1)
    if cv2.getWindowProperty("Face Detection with Landmarks", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.getWindowProperty("Stats and Sketch", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed manually. Exiting...")
        break

# Release resources
face_mesh.close()
video_capture.release()
cv2.destroyAllWindows()
