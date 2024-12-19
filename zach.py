import cv2

cap = cv2.VideoCapture(0)
face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    status, photo = cap.read()
    face_cor = face_model.detectMultiScale(photo)
    if len(face_cor) == 0:
        pass
    else:
        x1, y1, w, h = face_cor[0]
        x2, y2 = x1 + w, y1 + h
        photo = cv2.rectangle(photo, (x1, y1), (x2, y2), [0, 255, 0], 3)

    cv2.imshow("Face Detection", photo)
    if cv2.waitKey(10) == 13:
        break

cv2.destroyAllWindows()