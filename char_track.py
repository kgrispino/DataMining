import cv2
import numpy as np
from collections import deque
#https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

def ero_dilate(img_frame,low, high, kernel, kernel_2):
    mask = cv2.inRange(img_frame, low, high)

    erosion_mask = cv2.erode(mask, kernel, iterations=2)
    dilation_mask = cv2.dilate(erosion_mask, kernel_2, iterations=5)
    return dilation_mask

def find_contors(dilation_mask, img, tag):
    CONTOURS, _ = cv2.findContours(dilation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for j, CONTOURS in enumerate(CONTOURS):
        bBox = cv2.boundingRect(CONTOURS)
        contour_mask = np.zeros_like(dilation_mask)
        cv2.drawContours(contour_mask, CONTOURS, j, 255, -1)

        # Black screen with character
        # result = cv2.bitwise_and(frame, frame, mask=contour_mask)

        top_left, bottom_right = (bBox[0], bBox[1]), (bBox[0] + bBox[2], bBox[1] + bBox[3])
        cv2.putText(img, tag, (top_left[0] - 10, top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 10)
        rect_center = (int((top_left[0] + bottom_right[0])/2), int((top_left[1] + bottom_right[1])/2))
        #cv2.circle(img, rect_center, 5, (255, 255, 0), -1)
        # Black screen with character
        # cv2.rectangle(result, top_left, bottom_right, (0, 0, 255), 10)
        pts.appendleft(rect_center)
        #print(pts)
        for i in range(0, len(pts)):
            cv2.circle(img, pts[i], 5, (255, 255, 0), -1)

        #last_known_point = (pts[0][0], pts[0][1])
        #second_last_known_point = (pts[1][0], pts[1][1])



cap = cv2.VideoCapture('un.m4v')
buffer = 1000
counter = 0
pts = deque(maxlen=buffer)


while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hsv_frame = cv2.medianBlur(hsv_frame, 5)

    kernel = np.ones((7,7), np.uint8)
    kernel_2= np.ones((13, 13), np.uint8)

    kernel_skin = np.ones((3, 3), np.uint8)
    kernel_2_skin = np.ones((3, 3), np.uint8)

    low_skin = np.array([0,38,210])
    high_skin = np.array([93,139,255])
    low_redHair = np.array([137,153,110])
    high_redHair = np.array([183,194,187])
    low_blueHair = np.array([94, 167, 239])
    high_blueHair = np.array([103, 188, 255])

    blackScreen = np.array([0,0,0])

    redHair_mask = cv2.inRange(hsv_frame, low_redHair, high_redHair)
    blueHair_mask = cv2.inRange(hsv_frame, low_blueHair, high_blueHair)
    skin_mask = cv2.inRange(hsv_frame, low_blueHair, high_blueHair)
    blackScreen_mask = cv2.inRange(hsv_frame, low_blueHair, high_blueHair)

    skin_dilation = ero_dilate(hsv_frame, low_skin, high_skin, kernel_skin, kernel_2_skin)
    redHair_dilation = ero_dilate(hsv_frame, low_redHair, high_redHair, kernel, kernel_2)
    blueHair_dilation = ero_dilate(hsv_frame, low_blueHair, high_blueHair, kernel, kernel_2)

    #Good one
    redHair_skinMask = cv2.bitwise_or(skin_dilation, skin_dilation, mask=redHair_dilation)
    redHair_skinMask = cv2.dilate(redHair_skinMask, kernel_2, iterations=5)

    blueHair_skinMask = cv2.bitwise_and(skin_dilation, skin_dilation, mask=blueHair_dilation)
    blueHair_skinMask = cv2.dilate(blueHair_skinMask, kernel_2, iterations=5)
    #redHair_color = cv2.bitwise_and(frame, frame, mask=redHair_dilation)
    #skin_color = cv2.bitwise_and(frame, frame, mask=skin_dilation)
    #blueHair_color = cv2.bitwise_and(frame, frame, mask=blueHair_dilation)

    #redHair_skinMask = cv2.inRange(redHair_color, low_skin, high_skin)
    #blueHair_mask = cv2.inRange(hsv_frame, low_blueHair, high_blueHair)

    if (np.count_nonzero(gray_frame) < 94315):
        print("Black Frame")
        pts = deque(maxlen=buffer)

    find_contors(redHair_skinMask, frame, "Red Hair")
    find_contors(blueHair_skinMask, frame, "Blue Hair")

    #cv2.imwrite("redHair_color.png", redHair_color)
    cv2.imshow("Frame", frame)
    cv2.imshow("gray_frame", gray_frame)
    cv2.imshow("Red", redHair_skinMask)
    #cv2.imshow("Red", skin)

    key = cv2.waitKey(1)
    if key == 27:
        break