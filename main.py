import math
import random

import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all snake points
        self.length = []  # distance between points
        self.currentLength = 0  # total snake length
        self.allowedLength = 150  # total allowed length
        self.previousHead = 0, 0  # previous head point

        self.imageFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imageFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()
        self.score = 0
        self.game_over = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 600), random.randint(100, 600)

    def update(self, imgMain, currentHead):

        if self.game_over:
            cvzone.putTextRect(imgMain, "Game Over", [200, 200], scale=2, thickness=4, offset=10)
            cvzone.putTextRect(imgMain, f"Your Score: {self.score}", [200, 350], scale=2, thickness=4, offset=10)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.length.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # length reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.length):
                    self.currentLength -= length
                    self.length.pop(i)
                    self.points.pop(i)

                    if self.currentLength < self.allowedLength:
                        break

            # check if snake ate the food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and\
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(self.score)

            # draw snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 15)
                cv2.circle(img, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

            # draw food
            imgMain = cvzone.overlayPNG(imgMain, self.imageFood, (rx - self.wFood // 2, ry - self.hFood // 2))

            cvzone.putTextRect(imgMain, f"Score: {self.score}", [50, 80], scale=3, thickness=3, offset=10)

            # check fore collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            minDistance = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -0.1 <= minDistance <= 0.1:
                self.game_over = True
                self.points = []  # all snake points
                self.length = []  # distance between points
                self.currentLength = 0  # total snake length
                self.allowedLength = 150  # total allowed length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()
                self.score = 0

        return imgMain


game = SnakeGameClass("python1.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("r"):
        game.game_over = False
