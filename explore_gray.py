import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient

client = AdbClient()
device = client.devices()[0]
ratio = 1

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)

def on_thresh_trackbar(val):
    image_threshold = np.zeros_like(grayimage)
    image_threshold[grayimage > val] = 255
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))

image = get_screencap(device)
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('original')
cv2.namedWindow('trackbarWindow')
cv2.namedWindow('thresholded')
cv2.createTrackbar('low', 'trackbarWindow', 255,
                   255, on_thresh_trackbar)
cv2.imshow('original', cv2.resize(image, None, fx=ratio, fy=ratio))

cv2.waitKey()
