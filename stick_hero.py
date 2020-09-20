import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import time


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

while True:
    image = get_screencap(device)
    line = image[1365, 2:]
    target = np.nonzero(np.all(line == (27, 27, 247), axis=1))[0][0]
    if np.all(line[0] == 0):
        start = np.nonzero(np.all(line != (0, 0, 0), axis=1))[0][0]
    else:
        startpillar = np.nonzero(np.all(line == (0, 0, 0), axis=1))[0][0]
        start = np.nonzero(np.all(line != (0, 0, 0), axis=1))[0]
        start = start[np.nonzero(start > startpillar)[0][0]]

    #cv2.line(image, (start, 0), (start, 1920), (0, 255, 0), 3)
    # cv2.line(image,(target,0),(target,1920),(0,255,0),3)
    #cv2.imshow('image', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    device.input_swipe(500, 500, 505, 505, target-start)
    time.sleep(3)
