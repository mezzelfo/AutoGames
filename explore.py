import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient

max_value = 255
max_value_H = 360//2
low_H = 122
low_S = 40
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

client = AdbClient()
device = client.devices()[0]
ratio = 1

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)

image = get_screencap(device)

image_HSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    image_threshold = cv2.inRange(
        image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    image_threshold = cv2.inRange(
        image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    image_threshold = cv2.inRange(
        image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    image_threshold = cv2.inRange(
        image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    image_threshold = cv2.inRange(
        image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    image_threshold = cv2.inRange(
        image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    cv2.imshow('thresholded',  cv2.resize(image_threshold, None, fx=ratio, fy=ratio))


cv2.namedWindow('original')
cv2.namedWindow('trackbarWindow')
cv2.namedWindow('thresholded')
cv2.createTrackbar(low_H_name, 'trackbarWindow', low_H,
                   max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, 'trackbarWindow', high_H,
                   max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, 'trackbarWindow', low_S,
                   max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, 'trackbarWindow', high_S,
                   max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, 'trackbarWindow', low_V,
                   max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, 'trackbarWindow', high_V,
                   max_value, on_high_V_thresh_trackbar)

cv2.imshow('original', cv2.resize(image, None, fx=ratio, fy=ratio))

cv2.waitKey()
