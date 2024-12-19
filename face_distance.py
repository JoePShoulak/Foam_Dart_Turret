import cv2

def tuple_midpoint(t1, t2):
    (x1, y1), (x2, y2) = t1, t2

    return (abs(x1 + x2) / 2, abs(y1 + y2) / 2)

def tuple_distance(t1, t2):
    (x1, y1), (x2, y2) = t1, t2

    return pow((x2 - x1)**2 + (y2 - y1)**2, 0.5)

def get_eye_center(landmark):
    left_eye_outer = landmark[0]
    left_eye_inner = landmark[3]

    return tuple_midpoint(left_eye_outer, left_eye_inner)

def get_ipd(landmarks):
    left_eye_center = get_eye_center(landmarks["left_eye"])
    right_eye_center = get_eye_center(landmarks["right_eye"])
    
    return tuple_distance(left_eye_center, right_eye_center)

def draw_face_rectangle_and_label(frame, face_location, landmarks, scale_factor):
    # Scale face location back up
    top, right, bottom, left = face_location
    top *= scale_factor
    right *= scale_factor
    bottom *= scale_factor
    left *= scale_factor

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    # Get the eye distance (IPD)
    eye_distance = get_ipd(landmarks) * scale_factor

    # Draw a filled rectangle for the label
    label = f"IPD: {eye_distance:.2f}px"
    label_height = 30
    cv2.rectangle(frame, (left, bottom), (right, bottom + label_height), (0, 0, 255), cv2.FILLED)

    # Put white text in the label rectangle
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, label, (left + 6, bottom + label_height - 8), font, 0.6, (255, 255, 255), 1)
