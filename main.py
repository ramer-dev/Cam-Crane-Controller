import cmath

import cv2
import mediapipe as mp
import argparse
import numpy as np
from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand, MediaPipeFace
from utils_joint_angle import GestureRecognition
import socket
import time
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

HOST = '127.0.0.1'
PORT = 8000


class HandRecognition:
    def __init__(self):

        self.hands = MediaPipeHand(static_image_mode=False, max_num_hands=2)
        self.disp = DisplayHand(max_num_hands=2)
        self.cap = cv2.VideoCapture(0)
        self.gest = GestureRecognition(mode='eval')
        self.face_detection = MediaPipeFace(static_image_mode=False)
        self.client_socket = None

        self.img = self.cap.read()
        self.before_gesture = ""
        self.inner = False
        self.count_enable = True
        self.token = False
        self.counter = 0
        self.pose = None
        self.direction = 'out'
        self.direction_before = 'out'
        # 45 = camera fps(15fps) * 4 (sec)
        self.time_limit = 60

    def socketServer(self, host, port):

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen()
        self.client_socket, addr = server_socket.accept()
        print('Connected by', addr)

    def isInside(self, size: tuple, center: tuple, q: tuple):
        lt = np.array([center[0] - size[0], center[1] - size[1]])
        lb = np.array([center[0] - size[0], center[1] + size[1]])
        rt = np.array([center[0] + size[0], center[1] - size[1]])
        rb = np.array([center[0] + size[0], center[1] + size[1]])
        c = np.array(center)
        q = np.array(q)

        # Rule, length calculation must on clockwise direction.
        self.direction = "out"
        if (0 < np.dot(np.cross(rt - lt, q - lt), np.cross(q - lt, lt - rt)) and
                0 < np.dot(np.cross(c - rt, q - rt), np.cross(q - rt, lt - rt)) and
                0 < np.dot(np.cross(lt - c, q - c), np.cross(q - c, rt - c))):
            self.direction = "up"
        elif (0 < np.dot(np.cross(lt - lb, q - lb), np.cross(q - lb, lb - lt)) and
              0 < np.dot(np.cross(c - lt, q - lt), np.cross(q - lt, lb - lt)) and
              0 < np.dot(np.cross(lb - c, q - c), np.cross(q - c, lt - c))):
            self.direction = "left"

        elif (0 < np.dot(np.cross(lb - rb, q - rb), np.cross(q - rb, rb - lb)) and
              0 < np.dot(np.cross(c - lb, q - lb), np.cross(q - lb, rb - lb)) and
              0 < np.dot(np.cross(rb - c, q - c), np.cross(q - c, lb - c))):
            self.direction = "down"

        elif (0 < np.dot(np.cross(rb - rt, q - rt), np.cross(q - rt, rt - rb)) and
              0 < np.dot(np.cross(c - rb, q - rb), np.cross(q - rb, rt - rb)) and
              0 < np.dot(np.cross(rt - c, q - c), np.cross(q - c, rb - c))):
            self.direction = "right"
        self.img = cv2.putText(self.img, self.direction, (center[0], 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return self.direction

    def run(self):

        while self.cap.isOpened():
            ret, self.img = self.cap.read()
            if not ret:
                break

            # Flip image for 3rd person view
            self.img = cv2.flip(self.img, 1)

            # To improve performance, optionally mark image as not writeable to pass by reference
            self.img.flags.writeable = False

            # Feedforward to extract keypoint
            param = self.hands.forward(self.img)
            # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            evaluate = self.gest.eval(param[0]['angle'])
            if evaluate != self.pose:
                self.token = True
            else:
                self.token = False

            self.pose = evaluate
            if param[0]['class'] is not None:
                param[0]['gesture'] = self.gest.eval(param[0]['angle'])

            # UI

            rec_size = (180, 180)
            rec_center = (int(self.img.shape[1] / 2), int(self.img.shape[0] / 2))
            rec_left_top = ((rec_center[0] - rec_size[0]), (rec_center[1] - rec_size[1]))
            rec_bot_right = ((rec_center[0] + rec_size[0]), (rec_center[1] + rec_size[1]))

            self.img = cv2.rectangle(self.img, rec_left_top, rec_bot_right, (255, 0, 0), 2)

            self.img = cv2.line(self.img, rec_left_top, rec_bot_right, (255, 0, 0), 2)

            self.img = cv2.line(self.img, (rec_bot_right[0], rec_left_top[1]), (rec_left_top[0], rec_bot_right[1]),
                                (255, 0, 0), 2)

            # evaluate
            finger_tip = param[0]['keypt'][8]

            if self.pose == 'one':

                # if rec_bot_right[0] > finger_tip[0] > rec_left_top[0] \
                #         and rec_bot_right[1] > finger_tip[1] > rec_left_top[1]:
                #     self.inner = True
                # else:
                #     self.inner = False
                self.isInside(rec_size, rec_center, finger_tip)
                if self.direction != self.direction_before:
                    self.token = True
                else:
                    self.token = False

            elif self.pose == 'fist':
                hand_center = param[0]['keypt'][9]
                if rec_bot_right[0] > hand_center[0] > rec_left_top[0] \
                        and rec_bot_right[1] > hand_center[1] > rec_left_top[1]:

                    self.counter += 1
                    self.img = cv2.putText(self.img, str(int(self.counter*100 / self.time_limit))+"%", (rec_center[0] - 50, 200),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if self.counter > self.time_limit:
                        self.token = True
                        self.img = cv2.putText(self.img, "crain down", (rec_center[0] - 250, 200),
                                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                    else:
                        self.token = False

                else:
                    self.counter = 0
            else:
                self.counter = 0

            if self.token:
                if self.pose == 'one':
                    self.client_socket.sendall(self.direction.encode())
                    print(self.direction)
                elif self.pose == 'fist':
                    self.client_socket.sendall("crane".encode())
                    print("crane down")
                    self.counter = 0

            self.direction_before = self.direction
            self.img.flags.writeable = True

            # Display keypoint

            cv2.imshow('self.img 2D', self.disp.draw2d(self.img.copy(), param))

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == 32 and (param[0]['class'] is not None):
                cv2.waitKey(0)  # Pause display until user press any key
                self.__del__()

    def __del__(self):
        self.hands.pipe.close()
        self.cap.release()


hr = HandRecognition()
hr.socketServer(HOST, PORT)

hr.run()