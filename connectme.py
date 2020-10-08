import numpy as np
import cv2
from ppadb.client import Client as AdbClient
import matplotlib.pyplot as plt


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def pixel2type(hsvpixel):
    h, _, _ = hsvpixel
    if h == 64:
        return 'movable'
    elif h == 100:
        return 'rotable'
    elif h == 106:
        return 'empty'
    elif h == 173:
        return 'fixed'
    else:
        raise RuntimeError(f'hsvpixel {hsvpixel} non riconosciuto')


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)

image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
connect_mask = cv2.inRange(image_HSV, (0, 0, 0), (10, 10, 255))

s = 2**8-2
d = 15
for a in range(4):
    for b in range(4):
        x, y = 32+a*s, 460+b*s
        blocktype = pixel2type(image_HSV[y+s//3, x+s//3])
        if blocktype != 'empty':
            left = connect_mask[y+d:y-d+s, x+d]
            top = connect_mask[y+d, x+d:x-d+s]
            right = connect_mask[y+d:y-d+s, x-d+s]
            bottom = connect_mask[y-d+s, x+d:x-d+s]
            connections = [int(round(np.sum(l)/4080))
                           for l in [left, top, right, bottom]]
            print((a, b), (x+s//2, y+s//2), blocktype, connections)
