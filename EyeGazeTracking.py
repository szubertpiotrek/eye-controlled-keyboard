import cv2
import dlib
import numpy as np
from math import hypot

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
capture.set(cv2.CAP_PROP_FPS, 30)
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

        # hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 1)

        srodek_gora_lewe = srodek(landmarks.part(37), landmarks.part(38))
        srodek_dol_lewe = srodek(landmarks.part(41), landmarks.part(40))

        # ver_line = cv2.line(frame, srodek_gora_lewe, srodek_dol_lewe, (255, 0, 0), 1)

        left_point1 = (landmarks.part(42).x, landmarks.part(42).y)
        right_point1 = (landmarks.part(45).x, landmarks.part(45).y)

        # hor_line1 = cv2.line(frame, left_point1, right_point1, (255, 0, 0), 1)

        srodek_gora_prawe = srodek(landmarks.part(43), landmarks.part(44))
        srodek_dol_prawe = srodek(landmarks.part(47), landmarks.part(46))

        # ver_line1 = cv2.line(frame, srodek_gora_prawe, srodek_dol_prawe, (255, 0, 0), 1)

        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((srodek_gora_lewe[0] - srodek_dol_lewe[0]), (srodek_gora_lewe[1] - srodek_dol_lewe[1]))

        wsp_otwartosci_oka = hor_line_lenght / ver_line_lenght

        if wsp_otwartosci_oka < 6:
            cv2.putText(frame, "Otwarte", (50, 150), font, 3, (255, 0, 0))

        if wsp_otwartosci_oka > 6:
            cv2.putText(frame, "Zakmniete", (50, 150), font, 3, (255, 0, 0))

        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

        right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)

        height, width,_ = frame.shape
        mask = np.zeros((height, width), np.uint8)

        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray,gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])

        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _,threshold_eye = cv2.threshold(gray_eye,50,240,cv2.THRESH_BINARY)
        # _, contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     cv2.drawContours(threshold_eye, cnt, 0, (0, 255, 0), 3)
        # # gray_eye = cv2.medianBlur(gray_eye, 5)
        circles = cv2.HoughCircles(threshold_eye,cv2.HOUGH_GRADIENT,0.5, 41, param1=70, param2=30, minRadius=10,maxRadius=175)

        if circles is not None:
            for i in circles[0, :]:
                cv2.circle(threshold_eye, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(threshold_eye, (i[0], i[1]), 2, (0, 0, 255), 3)
        # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)


        cv2.imshow("Eye", threshold_eye)
        cv2.imshow("Eyess", gray_eye)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()