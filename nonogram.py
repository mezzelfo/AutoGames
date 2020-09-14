import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)
# cv2.imwrite('img.png',image)
#image = cv2.imread('img.png')

ratio = 1
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
numbers = cv2.inRange(imageHSV, (120, 40, 0), (180, 255, 255))
hor_numbers = image[340:545, 190:-15]
ver_numbers = image[550:-500, 15:190]

def rand_color():
    c = np.random.randint(0,255,3)
    return tuple([int(c[0]),int(c[1]),int(c[2])])

level = 10
w = int(120-4*level)
s = int(level*3/10)
for x in np.linspace(200, 1060, level, False):
    for d in range(1, s+1):
        cv2.rectangle(image, (int(x), 340), (int(x+w),
                                             int(340+d*200/s)), rand_color(), 5)

for y in np.linspace(550, 1420, level, False):
    for d in range(1, s+1):
        cv2.rectangle(image, (20, int(y)), (int(20+d*180/s),
                                            int(y+w)), rand_color(), 5)

cv2.imshow('original', cv2.resize(image, None, fx=ratio, fy=ratio))
# cv2.imshow('numbers',cv2.resize(numbers,None,fx=ratio,fy=ratio))
# cv2.imshow('hor_numbers',cv2.resize(hor_numbers,None,fx=ratio,fy=ratio))
# cv2.imshow('ver_numbers',cv2.resize(ver_numbers,None,fx=ratio,fy=ratio))
# cv2.imshow('zone',cv2.resize(zone,None,fx=ratio,fy=ratio))
cv2.waitKey()
