import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs if possible

import cv2
import mediapipe as mp
import math
import numpy as np
import sys

# Print initializing message
print("initializing")

# ANSI color codes for terminal output
BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"

# Colored square character (full block)
BLOCK_CHAR = "█"

# Landmark ID constants
LEFT_EYE_IDS = [33, 133, 159, 145]
RIGHT_EYE_IDS = [362, 263, 386, 374]
MOUTH_CORNER_IDS = [61, 291]

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam. Please check your device.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                  max_num_faces=1, 
                                  min_detection_confidence=0.5)

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

def print_stats_tabular(ipd, mouth_width, after_clear):
    """Print stats in a tabular form with color-coded squares to the terminal."""
    if after_clear:
        # Move cursor to the top-left corner, move down 2 lines past "Foam Dart Turret" and blank line
        print("\033[H", end="")
        print("\033[2E", end="")
    # If not after_clear, we simply do nothing special. We rely on initial logs until stable data.

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

after_clear = False
initial_iterations = 0
cleared_screen = False

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        ipd, mouth_width = None, None

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

                # Draw lines on the video frame
                cv2.line(frame, left_eye_pixel, right_eye_pixel, (255, 0, 0), 2)
                cv2.line(frame, left_mouth_pixel, right_mouth_pixel, (0, 255, 0), 2)

        if ipd is not None and mouth_width is not None:
            initial_iterations += 1
            # After a couple iterations, we assume logs have appeared and we can now clear
            if initial_iterations == 2 and not cleared_screen:
                # Clear and print title now that logs (if any) appeared
                print("\033c", end="")
                print("Foam Dart Turret\n")
                cleared_screen = True
                after_clear = True

            print_stats_tabular(ipd, mouth_width, after_clear)

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
