import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs if possible

import cv2
import mediapipe as mp
import math
import numpy as np
import sys
import argparse

########################################
# Constants
########################################
# ANSI and Character Constants
CSI = "\033["
CLEAR_SCREEN_HOME = CSI + "2J" + CSI + "H"  # Clear screen and move cursor home
CURSOR_HOME = CSI + "H"
DOWN_2 = CSI + "2E"

# Colors
BLUE = CSI + "34m"
GREEN = CSI + "32m"
RESET = CSI + "0m"

# Symbols
BLOCK_CHAR = "█"

# Landmark ID constants
LEFT_EYE_IDS = [33, 133, 159, 145]
RIGHT_EYE_IDS = [362, 263, 386, 374]
MOUTH_CORNER_IDS = [61, 291]

########################################

# Parse arguments
parser = argparse.ArgumentParser(description="Run face detection with IPD and mouth width, optionally show face mesh.")
parser.add_argument("--mesh", action="store_true", help="Draw the face mesh under the biometric lines")
args = parser.parse_args()

print("Initializing Mediapipe...", end="", flush=True)

# Initialize Mediapipe Drawing (for optional mesh)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

print("Done!")
print("Initializing Webcam...")

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam. Please check your device.")
    exit()
print("Done!")

# Clear the terminal screen after initialization
print(CLEAR_SCREEN_HOME, end="")
print("Foam Dart Turret\n")

def get_landmark_pixel(landmarks, frame_shape):
    return [
        (int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0]))
        for landmark in landmarks
    ]

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def get_eye_center(landmarks, frame_shape):
    eye_pixel = get_landmark_pixel(landmarks, frame_shape)
    center_x = sum(p[0] for p in eye_pixel) / len(eye_pixel)
    center_y = sum(p[1] for p in eye_pixel) / len(eye_pixel)
    return int(center_x), int(center_y)

def print_stats_tabular(ipd, mouth_width):
    """Print stats in a tabular form with color-coded squares to the terminal."""
    print(CURSOR_HOME, end="")
    print(DOWN_2, end="")

    data = [
        ("Interpupillary Distance (IPD):", f"{ipd:.2f} px", BLUE),
        ("Mouth Width:", f"{mouth_width:.2f} px", GREEN),
    ]

    max_label_len = max(len(label) for (label, value, color) in data)
    max_value_len = max(len(value) for (label, value, color) in data)
    spacing = 4

    for (label, value, color) in data:
        label_col = label.ljust(max_label_len)
        value_col = value.rjust(max_value_len)
        line_str = f"{color}{BLOCK_CHAR}{RESET} {label_col}{' ' * spacing}{value_col}"
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
                left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE_IDS]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE_IDS]

                left_eye_pixel = get_eye_center(left_eye_landmarks, frame.shape)
                right_eye_pixel = get_eye_center(right_eye_landmarks, frame.shape)
                ipd = calculate_distance(left_eye_pixel, right_eye_pixel)

                left_mouth_corner = face_landmarks.landmark[MOUTH_CORNER_IDS[0]]
                right_mouth_corner = face_landmarks.landmark[MOUTH_CORNER_IDS[1]]
                left_mouth_pixel, right_mouth_pixel = get_landmark_pixel([left_mouth_corner, right_mouth_corner], frame.shape)
                mouth_width = calculate_distance(left_mouth_pixel, right_mouth_pixel)

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
                # IPD line - blue
                cv2.line(frame, left_eye_pixel, right_eye_pixel, (255, 0, 0), 2)
                # Mouth line - green
                cv2.line(frame, left_mouth_pixel, right_mouth_pixel, (0, 255, 0), 2)

        if ipd and mouth_width:
            print_stats_tabular(ipd, mouth_width)

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
