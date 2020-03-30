import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('figure', figsize=(10, 5))

img = cv2.imread('cel.png',0)
img2 = img.copy()
template = cv2.imread('temp.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

cap = cv2.VideoCapture('un.m4v')

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > template.shape[0] and gray.shape[1] > template.shape[1]:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCORR)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        cv2.imshow('orginal', frame)
        cv2.imshow('template', res)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break