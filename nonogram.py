import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import pytesseract


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def see_numbers(block, vertical=False):
    data = pytesseract.image_to_boxes(
        block, config='--psm 6', output_type=pytesseract.Output.DICT)
    text = []
    build = data['char'][0]
    for i in range(1, len(data['char'])):
        if data['char'][i] in [' ', '\n']:
            continue
        if 3 < data['left'][i]-data['left'][i-1] < 16:
            build += data['char'][i]
        else:
            text.append(int(build))
            build = data['char'][i]
    text.append(int(build))
    return text


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)

ratio = 1
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
numbers = cv2.inRange(imageHSV, (120, 40, 0), (180, 255, 255))
numbers = 255-numbers

print('Start image analysing')
level = 20
w = int(120-4*level)
horizontal_numbers = []
for x in np.linspace(200, 1060, level, False):
    block = numbers[340:540, int(x):int(x+w)]
    horizontal_numbers.append(see_numbers(block))
vertical_numbers = []
for y in np.linspace(550, 1420, level, False):
    block = numbers[int(y):int(y+w), 20:180]
    vertical_numbers.append(see_numbers(block))

print('Start problem solving')
