import cv2
import dlib
import numpy as np

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while(True):
    _,frame = capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)
        landmarks = predictor(gray, face)

        for points in range(36, 48):
            x = landmarks.part(points).x
            y = landmarks.part(points).y
            cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)

        print(landmarks)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()