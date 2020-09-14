import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import math
import time

client = AdbClient()
device = client.devices()[0]
ratio = 1


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def calcPullback(start_hoop, end_hoop, accel=2872):
    x1, y1, _ = start_hoop
    x2, y2, a2 = end_hoop
    t = math.tan(a2)
    a = (t*(-x1+x2)+y1-y2)/(x1-x2)**2
    b = (a * (-x1**2+x2**2)+y1-y2)/(x1-x2)
    c = y1 - x1*(b+a*x1)
    xpeak = int(-b/(2*a))
    ypeak = int(a*xpeak**2+b*xpeak+c)

    delX = xpeak - x1
    delY = ypeak - y1
    yVel = -math.sqrt(-2 * accel * delY)
    airTime = -yVel/accel
    xVel = delX / airTime
    throwAngle = math.atan(xVel/yVel)
    yPullBack = delY / 1.7
    xPullBack = math.tan(throwAngle) * yPullBack * .8
    return xPullBack, yPullBack, airTime


def get_hoop_ellipse(hsvimg, hsvbottom, hsvtop, mode=cv2.RETR_EXTERNAL):
    inrange = cv2.inRange(hsvimg, hsvbottom, hsvtop)
    inrange[-150:] = 0
    inrange[:600] = 0
    cnts = cv2.findContours(inrange, mode, cv2.CHAIN_APPROX_SIMPLE)[0]
    hoop = max(cnts, key=cv2.contourArea)
    (x, y), _, alpha = cv2.fitEllipse(hoop)
    return (int(x), int(y), math.radians(-alpha))


def get_hoops(bgrimage, base_angle=65):
    hsvimg = cv2.cvtColor(bgrimage, cv2.COLOR_BGR2HSV)
    sx, sy, sangle = get_hoop_ellipse(hsvimg, (0, 0, 165), (180, 255, 175))
    ex, ey, enagle = get_hoop_ellipse(hsvimg, (0, 180, 215), (20, 220, 230))
    if sx > ex:
        enagle = max(math.radians(-base_angle), enagle)
    else:
        enagle = max(math.radians(base_angle+180), enagle)

    return ((sx, sy, sangle), (ex, ey, enagle))


def different_config(hoop1, hoop2, threshold=10):
    return any(abs(a-b) > threshold for a, b in zip(hoop1, hoop2))


def play():
    first_img = get_screencap(device)
    _, end_hoop1 = get_hoops(first_img)
    time.sleep(0.3)
    second_img = get_screencap(device)
    start_hoop2, end_hoop2 = get_hoops(second_img)

    if different_config(end_hoop1, end_hoop2):
        print('moving')
        dx, dy, airtime = calcPullback(start_hoop2, end_hoop2)
        start = time.time()
        while different_config(end_hoop1,end_hoop2):
            first_img = get_screencap(device)
            _, end_hoop1 = get_hoops(first_img)
        cycle_time = time.time()-start
        time.sleep(cycle_time-airtime)
    else:
        print('not moving')
    dx, dy, _ = calcPullback(start_hoop2, end_hoop2)
    device.input_swipe(700, 800, 700 - dx, 800 - dy, 500)


for _ in range(10):
    play()
