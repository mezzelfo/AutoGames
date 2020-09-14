import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import math

client = AdbClient()
device = client.devices()[0]
ratio = 0.35


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def calcPullback():
    accel = 2872
    delX = peak_pos[0] - sh_pos[0]
    delY = peak_pos[1] - sh_pos[1]
    yVel = -math.sqrt(-2 * accel * delY)
    airTime = -yVel/accel
    xVel = delX / airTime
    throwAngle = math.atan(xVel/yVel)
    yPullBack = delY / 1.7
    xPullBack = math.tan(throwAngle) * yPullBack * .8
    return xPullBack, yPullBack, airTime


def get_hoops_inrange(hsvimg, hsvbottom, hsvtop, mode=cv2.RETR_EXTERNAL):
    inrange = cv2.inRange(hsvimg, hsvbottom, hsvtop)
    inrange[-150:] = 0
    inrange[:600] = 0
    cnts = cv2.findContours(inrange, mode, cv2.CHAIN_APPROX_SIMPLE)[0]
    hoop = max(cnts, key=cv2.contourArea)
    (x, y), _, alpha = cv2.fitEllipse(hoop)
    return (int(x), int(y)), math.radians(-alpha)


img = get_screencap(device)
hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
sh_pos, sh_angle = get_hoops_inrange(hsvimg, (0, 0, 165), (180, 255, 175))
eh_pos, eh_angle = get_hoops_inrange(hsvimg, (0, 180, 215), (20, 220, 230))

base_angle = 70
if sh_pos[0] > eh_pos[0]:
    eh_angle = max(math.radians(-base_angle), eh_angle)
else:
    eh_angle = max(math.radians(base_angle+180), eh_angle)

x1, y1 = sh_pos
x2, y2 = eh_pos
t = math.tan(eh_angle)
a = (t*(-x1+x2)+y1-y2)/(x1-x2)**2
b = (a * (-x1**2+x2**2)+y1-y2)/(x1-x2)
c = y1 - x1*(b+a*x1)
xpeak = -b/(2*a)
ypeak = a*xpeak**2+b*xpeak+c
peak_pos = (int(xpeak), int(ypeak))
print(f'{sh_pos=}')
print(f'{eh_pos=}')
print(f'Parabola: {a=},{b=},{c=}')
print(f'Peak: {peak_pos}')

cv2.circle(img, sh_pos, 20, (255, 0, 0), -1)
cv2.circle(img, eh_pos, 20, (0, 255, 0), -1)
cv2.circle(img, peak_pos, 20, (0, 255, 255), -1)
for i in range(min(sh_pos[0], eh_pos[0]), max(sh_pos[0], eh_pos[0]), 5):
    cv2.circle(img, (i, int(a*i**2+b*i+c)), 1, (0, 0, 0), 2)
x1, y1 = eh_pos
x2 = x1 + 100 * math.cos(eh_angle)
y2 = y1 + 100 * math.sin(eh_angle)
cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)

dx, dy, _ = calcPullback()
device.input_swipe(700, 800, 700 - dx, 800 - dy, 500)

cv2.imshow('img', cv2.resize(img, None, fx=ratio, fy=ratio))
cv2.waitKey()
