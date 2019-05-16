from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from matplotlib import pyplot as plt
import dlib
from math import hypot


def srodek(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    xmin=0
    ymin=0
    xmax=0
    ymax=0
    print(detections)
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(3.07*x), float(1.73*y), float(3.07*w), float(1.73*h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img, xmin, ymin, xmax, ymax
    

netMain = None
metaMain = None
altNames = None

def eyeGazeTacking():

    global metaMain, netMain, altNames
    configPath = "cfg/yolov3-tiny-1c.cfg"
    weightPath = "backup/yolov3-tiny-1c_last.weights"
    metaPath = "data/eye.data"
   
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    kalman = cv2.KalmanFilter(2, 1, 0)

    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        stime = time.time()
        ret, image = cap.read()
        keyboard = np.zeros((560,1000,3), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

            # hor_line = cv2.line(image, left_point, right_point, (0, 0, 255), 3)

            srodek_gora_lewe = srodek(landmarks.part(37), landmarks.part(38))
            srodek_dol_lewe = srodek(landmarks.part(41), landmarks.part(40))

            # ver_line = cv2.line(image, srodek_gora_lewe, srodek_dol_lewe, (0, 0, 255), 3)

            left_point1 = (landmarks.part(42).x, landmarks.part(42).y)
            right_point1 = (landmarks.part(45).x, landmarks.part(45).y)

            # hor_line1 = cv2.line(image, left_point1, right_point1, (0, 0, 255), 3)

            srodek_gora_prawe = srodek(landmarks.part(43), landmarks.part(44))
            srodek_dol_prawe = srodek(landmarks.part(47), landmarks.part(46))

            # ver_line1 = cv2.line(image, srodek_gora_prawe, srodek_dol_prawe, (0, 0, 255), 3)

            hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
            ver_line_lenght = hypot((srodek_gora_lewe[0] - srodek_dol_lewe[0]),
                                    (srodek_gora_lewe[1] - srodek_dol_lewe[1]))

            wsp_otwartosci_oka = hor_line_lenght / ver_line_lenght

            if wsp_otwartosci_oka < 6:
                cv2.putText(image, "Otwarte", (50, 150), font, 3, (255, 0, 0))

            if wsp_otwartosci_oka > 6:
                cv2.putText(image, "Zakmniete", (50, 150), font, 3, (255, 0, 0))

            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            # cv2.polylines(image, [left_eye_region], True, (0, 0, 255), 2)

            right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                         (landmarks.part(37).x, landmarks.part(37).y),
                                         (landmarks.part(38).x, landmarks.part(38).y),
                                         (landmarks.part(39).x, landmarks.part(39).y),
                                         (landmarks.part(40).x, landmarks.part(40).y),
                                         (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            # cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)

            # height, width, _ = image.shape
            # mask = np.zeros((height, width), np.uint8)
            #
            # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            # cv2.polylines(mask, [left_eye_region], True, 255, 2)
            #
            # min_x = np.min(left_eye_region[:, 0])
            # max_x = np.max(left_eye_region[:, 0])
            #
            # min_y = np.min(left_eye_region[:, 1])
            # max_y = np.max(left_eye_region[:, 1])
            #
            # gray_eye = image[min_y: max_y, min_x: max_x]
            # _, threshold_eye = cv2.threshold(gray_eye, 50, 240, cv2.THRESH_BINARY)
            color_eye = image
            color_eye = cv2.cvtColor(color_eye, cv2.COLOR_BGR2RGB)
            color_eye = cv2.resize(color_eye,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, color_eye.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            _, xmin, ymin, xmax, ymax = cvDrawBoxes(detections, image)
            xsr = int(round(xmin + ((xmax-xmin)/2)))
            ysr = int(round(ymin + ((ymax-ymin)/2)))
            cv2.line(image, (int(round(left_point1[0])), int(round(left_point1[1]))), (int(round(right_point1[0])), int(round(right_point1[1]))), (0, 0, 255), 2)
            cv2.line(image, (int(round(srodek_gora_prawe[0])), int(round(srodek_gora_prawe[1]))), (int(round(srodek_dol_prawe[0])),int(round(srodek_dol_prawe[1]))), (0, 0, 255), 2)
            cv2.circle(image, (xsr, ysr), 4, (255, 0, 0), 4)
            # color_eye = cv2.cvtColor(color_eye, cv2.COLOR_BGR2RGB)
            print(ver_line_lenght)
            print(hor_line_lenght)
            # proportial_ratio_x = 1000/1280*15
            # proportial_ratio_y = 560/720*30
            # cv2.line(keyboard, (int(round((srodek_gora_prawe[0]-left_point1[0])*proportial_ratio_x)), int(round((srodek_gora_prawe[1]-srodek_gora_prawe[1])*proportial_ratio_y))),
            #          (int(round((srodek_dol_prawe[0]-left_point1[0])*proportial_ratio_x)), int(round((srodek_dol_prawe[1]-srodek_gora_prawe[1])*proportial_ratio_y))), (0, 0, 255), 2)
            # cv2.line(keyboard, (int((round(left_point1[0]-left_point1[0])*proportial_ratio_x)), int(round((left_point1[1]-srodek_gora_prawe[1])*proportial_ratio_y))),
            #          (int(round((right_point1[0]-left_point1[0])*proportial_ratio_x)), int(round((right_point1[1]-srodek_gora_prawe[1])*proportial_ratio_y))), (0, 0, 255), 2)
            #
            # cv2.circle(keyboard, (int(round((xsr-left_point1[0])*proportial_ratio_x)), int(round((ysr-srodek_gora_prawe[1])*proportial_ratio_y))), 4, (255, 0, 0), 4)

            proportial_ratio_x = hor_line_lenght/1000
            proportial_ratio_y = ver_line_lenght/560
            cv2.line(keyboard, (int(round((srodek_gora_prawe[0] - left_point1[0])/proportial_ratio_x)),
                                int(round((srodek_gora_prawe[1] - srodek_gora_prawe[1])/proportial_ratio_y))),
                     (int(round((srodek_dol_prawe[0] - left_point1[0])/proportial_ratio_x)),
                      int(round((srodek_dol_prawe[1] - srodek_gora_prawe[1])/proportial_ratio_y))), (0, 0, 255), 2)
            cv2.line(keyboard, (int((round(left_point1[0] - left_point1[0])/proportial_ratio_x)),
                                int(round((left_point1[1] - srodek_gora_prawe[1])/proportial_ratio_y))),
                     (int(round((right_point1[0] - left_point1[0])/proportial_ratio_x)),
                      int(round((right_point1[1] - srodek_gora_prawe[1])/proportial_ratio_y))), (0, 0, 255), 2)

            cv2.circle(keyboard, (int(round((xsr- left_point1[0])/proportial_ratio_x)),
                                  int(round((ysr - srodek_gora_prawe[1])/proportial_ratio_y))), 4, (255, 0, 0), 4)

        cv2.imshow("Eyess", image)
        cv2.imshow("Keyboard", keyboard)

        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    eyeGazeTacking()








