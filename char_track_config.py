import cv2
import cv2 as cv
import numpy as np
import time


def ero_dilate(img_frame, low, high):
    mask = cv2.inRange(img_frame, low, high)

    erosion_mask = cv2.erode(mask, kernel, iterations=1)
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
        # Black screen with character
        # cv2.rectangle(result, top_left, bottom_right, (0, 0, 255), 10)


cap = cv2.VideoCapture('un.m4v')

uh = 93
us = 139
uv = 255
lh = 0
ls = 38
lv = 210
lower_hsv = np.array([lh, ls, lv])
upper_hsv = np.array([uh, us, uv])

window_name = "HSV Calibrator"
cv.namedWindow(window_name)


def nothing(x):
    print("Trackbar value: " + str(x))
    pass


# create trackbars for Upper HSV
cv.createTrackbar('UpperH', window_name, 0, 255, nothing)
cv.setTrackbarPos('UpperH', window_name, uh)

cv.createTrackbar('UpperS', window_name, 0, 255, nothing)
cv.setTrackbarPos('UpperS', window_name, us)

cv.createTrackbar('UpperV', window_name, 0, 255, nothing)
cv.setTrackbarPos('UpperV', window_name, uv)

# create trackbars for Lower HSV
cv.createTrackbar('LowerH', window_name, 0, 255, nothing)
cv.setTrackbarPos('LowerH', window_name, lh)

cv.createTrackbar('LowerS', window_name, 0, 255, nothing)
cv.setTrackbarPos('LowerS', window_name, ls)

cv.createTrackbar('LowerV', window_name, 0, 255, nothing)
cv.setTrackbarPos('LowerV', window_name, lv)

font = cv.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv_frame = cv2.medianBlur(hsv_frame, 5)

    kernel = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((13, 13), np.uint8)

    low_skin = np.array([0, 71, 212])
    high_skin = np.array([26, 150, 255])

    low_redHair = np.array([137, 153, 110])
    high_redHair = np.array([183, 194, 187])

    low_blueHair = np.array([94, 167, 239])
    high_blueHair = np.array([103, 188, 255])

    mask = cv.inRange(hsv_frame, lower_hsv, upper_hsv)
    cv.putText(mask, 'Lower HSV: [' + str(lh) + ',' + str(ls) + ',' + str(lv) + ']', (10, 30), font, 0.5,
               (200, 255, 155), 1, cv.LINE_AA)
    cv.putText(mask, 'Upper HSV: [' + str(uh) + ',' + str(us) + ',' + str(uv) + ']', (10, 60), font, 0.5,
               (200, 255, 155), 1, cv.LINE_AA)

    cv.imshow(window_name, mask)
    cv.imwrite("frame.png", frame)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper HSV trackbars
    uh = cv.getTrackbarPos('UpperH', window_name)
    us = cv.getTrackbarPos('UpperS', window_name)
    uv = cv.getTrackbarPos('UpperV', window_name)
    upper_blue = np.array([uh, us, uv])
    # get current positions of Lower HSCV trackbars
    lh = cv.getTrackbarPos('LowerH', window_name)
    ls = cv.getTrackbarPos('LowerS', window_name)
    lv = cv.getTrackbarPos('LowerV', window_name)
    upper_hsv = np.array([uh, us, uv])
    lower_hsv = np.array([lh, ls, lv])

    time.sleep(.1)

cv.destroyAllWindows()