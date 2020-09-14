import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient

client = AdbClient()
device = client.devices()[0]
ratio = 0.6

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)

image = get_screencap(device)
backimage = image.copy()
grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
grayscale[:170,:] = 0
grayscale[:,:130] = 0
grayscale[:,-130:] = 0
cv2.imshow('gray',cv2.resize(grayscale,None,fx=ratio,fy=ratio))

param1 = 50
param2 = 100



def ontrack1(val):
    global param1
    param1 = max(val,1)
    show()

def ontrack2(val):
    global param2
    param2 = max(val,1)
    show()
    
def show():
    global image, param1, param2, backimage, grayscale
    cv2.setTrackbarPos('track1','traks',param1)
    cv2.setTrackbarPos('track2','traks',param2)
    circles = cv2.HoughCircles(grayscale,cv2.HOUGH_GRADIENT,1.1,15,None,param1,param2,18,26)
    image = backimage.copy()
    if circles is not None:
        circles = circles[0]
        print(len(circles))
        #print(circles)
        for x,y,r in circles:
            image = cv2.circle(image,(int(x),int(y)),int(r),(0,0,255),5)
    else:
        print('None')
    cv2.imshow('img',cv2.resize(image,None,fx=ratio,fy=ratio))


cv2.namedWindow('img')
cv2.namedWindow('traks')

cv2.createTrackbar('track1','traks',1,255,ontrack1)
cv2.createTrackbar('track2','traks',1,255,ontrack2)

ontrack1(62)
ontrack2(32)

cv2.waitKey()