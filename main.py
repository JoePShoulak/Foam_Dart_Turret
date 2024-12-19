import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs if possible

import sys
import cv2
import mediapipe as mp
import math
import argparse

########################################
# Constants
########################################
# ANSI and Character Constants
CSI = "\033["
CLEAR_SCREEN_HOME = CSI + "2J" + CSI + "H"  # Clear screen and move cursor home
CURSOR_HOME = CSI + "H"
DOWN_2 = CSI + "2E"

# Terminal Text Colors
BLUE_TEXT = CSI + "34m"
GREEN_TEXT = CSI + "32m"
YELLOW_TEXT = CSI + "33m"
RESET_TEXT = CSI + "0m"

# Video Line Colors
BLUE_RGB = (255, 0, 0)  # Blue line for IPD
GREEN_RGB = (0, 255, 0)  # Green line for Mouth Width
YELLOW_RGB = (0, 255, 255)  # Yellow line for Nose-to-Chin

# Symbols
BLOCK_CHAR = "█"

# Landmark ID constants
LEFT_EYE_IDS = [33, 133, 159, 145]
RIGHT_EYE_IDS = [362, 263, 386, 374]
MOUTH_CORNER_IDS = [61, 291]
NOSE_TIP_ID = 1
CHIN_ID = 152

########################################

# Parse arguments
parser = argparse.ArgumentParser(description="Run face detection with IPD, mouth width, and nose-to-chin distance, optionally show face mesh.")
parser.add_argument("--mesh", action="store_true", help="Draw the face mesh under the biometric lines")
args = parser.parse_args()

print("Initializing Mediapipe and Webcam...", end="", flush=True)
# Suppress Mediapipe logs by redirecting stderr
sys.stderr = open(os.devnull, 'w')

# Initialize Mediapipe Drawing (for optional mesh)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# Restore stderr after initialization
sys.stderr.close()
sys.stderr = sys.__stderr__

# Initialize Webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam. Please check your device.")
    exit()
print("Done!")

# Clear screen state variables
after_clear = False
initial_iterations = 0
cleared_screen = False

# Clear the terminal screen after initialization
print(CLEAR_SCREEN_HOME, end="")
print("Foam Dart Turret\n")

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def get_landmark_pixel(landmarks, frame_shape):
    """Convert normalized landmarks to pixel coordinates."""
    return [
        (int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0]))
        for landmark in landmarks
    ]

def get_eye_center(landmarks, frame_shape):
    """Calculate the center of the eye from landmarks."""
    eye_pixel = get_landmark_pixel(landmarks, frame_shape)
    center_x = sum(p[0] for p in eye_pixel) / len(eye_pixel)
    center_y = sum(p[1] for p in eye_pixel) / len(eye_pixel)
    return int(center_x), int(center_y)

def print_stats_tabular(ipd, mouth_width, nose_chin_distance):
    """Print stats in a tabular form with color-coded squares to the terminal."""
    print(CURSOR_HOME, end="")
    print(DOWN_2, end="")

    data = [
        ("Interpupillary Distance (IPD):", f"{ipd:.2f} px", BLUE_TEXT),
        ("Mouth Width:", f"{mouth_width:.2f} px", GREEN_TEXT),
        ("Nose-to-Chin Distance:", f"{nose_chin_distance:.2f} px", YELLOW_TEXT),
    ]

    max_label_len = max(len(label) for (label, value, color) in data)
    max_value_len = max(len(value) for (label, value, color) in data)
    spacing = 4

    for (label, value, color) in data:
        label_col = label.ljust(max_label_len)
        value_col = value.rjust(max_value_len)
        line_str = f"{color}{BLOCK_CHAR}{RESET_TEXT} {label_col}{' ' * spacing}{value_col}"
        print(line_str)
    print()

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE_IDS]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE_IDS]

                left_eye_pixel = get_eye_center(left_eye_landmarks, frame.shape)
                right_eye_pixel = get_eye_center(right_eye_landmarks, frame.shape)
                ipd = calculate_distance(left_eye_pixel, right_eye_pixel)

                # Extract mouth landmarks
                left_mouth_corner = face_landmarks.landmark[MOUTH_CORNER_IDS[0]]
                right_mouth_corner = face_landmarks.landmark[MOUTH_CORNER_IDS[1]]
                left_mouth_pixel, right_mouth_pixel = get_landmark_pixel(
                    [left_mouth_corner, right_mouth_corner], frame.shape
                )
                mouth_width = calculate_distance(left_mouth_pixel, right_mouth_pixel)

                # Extract nose-to-chin landmarks
                nose_tip = face_landmarks.landmark[NOSE_TIP_ID]
                chin = face_landmarks.landmark[CHIN_ID]
                nose_pixel, chin_pixel = get_landmark_pixel([nose_tip, chin], frame.shape)
                nose_chin_distance = calculate_distance(nose_pixel, chin_pixel)

                # If --mesh is enabled, draw the face mesh underneath
                if args.mesh:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                # Draw lines on the video frame
                cv2.line(frame, left_eye_pixel, right_eye_pixel, BLUE_RGB, 2)  # IPD line
                cv2.line(frame, left_mouth_pixel, right_mouth_pixel, GREEN_RGB, 2)  # Mouth Width line
                cv2.line(frame, nose_pixel, chin_pixel, YELLOW_RGB, 2)  # Nose-to-Chin line

        if ipd and mouth_width and nose_chin_distance:
            initial_iterations += 1
            # After a couple iterations, we assume logs have appeared and we can now clear
            if initial_iterations == 2 and not cleared_screen:
                # Clear and print title now that logs hopefully won't appear anymore
                print(CLEAR_SCREEN_HOME, end="")
                print("Foam Dart Turret\n")
                cleared_screen = True
                after_clear = True

            print_stats_tabular(ipd, mouth_width, nose_chin_distance)

        cv2.imshow("Face Detection with Landmarks", frame)
        key = cv2.waitKey(1)
        if cv2.getWindowProperty("Face Detection with Landmarks", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed manually. Exiting...")
            break
except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Exiting gracefully...")

# Cleanup
face_mesh.close()
video_capture.release()
cv2.destroyAllWindows()
