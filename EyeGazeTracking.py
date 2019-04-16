import cv2
import dlib
import numpy as np
from math import hypot

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def srodek(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_SIMPLEX

while (True):
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)
        landmarks = predictor(gray, face)

        # for points in range(36, 48):
        #   x = landmarks.part(points).x
        #  y = landmarks.part(points).y
        # cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)

        hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 1)

        srodek_gora_lewe = srodek(landmarks.part(37), landmarks.part(38))
        srodek_dol_lewe = srodek(landmarks.part(41), landmarks.part(40))

        ver_line = cv2.line(frame, srodek_gora_lewe, srodek_dol_lewe, (255, 0, 0), 1)

        left_point1 = (landmarks.part(42).x, landmarks.part(42).y)
        right_point1 = (landmarks.part(45).x, landmarks.part(45).y)

        hor_line1 = cv2.line(frame, left_point1, right_point1, (255, 0, 0), 1)

        srodek_gora_prawe = srodek(landmarks.part(43), landmarks.part(44))
        srodek_dol_prawe = srodek(landmarks.part(47), landmarks.part(46))

        ver_line1 = cv2.line(frame, srodek_gora_prawe, srodek_dol_prawe, (255, 0, 0), 1)

        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((srodek_gora_lewe[0] - srodek_dol_lewe[0]), (srodek_gora_lewe[1] - srodek_dol_lewe[1]))

        wsp_otwartosci_oka = hor_line_lenght / ver_line_lenght

        if wsp_otwartosci_oka < 6:
            cv2.putText(frame, "Otwarte", (50, 150), font, 3, (255, 0, 0))

        if wsp_otwartosci_oka > 6:
            cv2.putText(frame, "Zakmniete", (50, 150), font, 3, (255, 0, 0))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()