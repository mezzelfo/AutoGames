import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
from itertools import product
from time import sleep
from math import floor

new_level_button = [(375, 880, '4x4'), (536, 880, '5x5'), (734, 880, '6x6'),
                    (375, 1090, '7x7'), (536, 1090, '8x8'), (734, 1090, '9x9')]

numbers_templates = np.load('oh_no_numbers.npy')


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def get_almost_unique(vec, threshold=20):
    return np.delete(np.unique(vec), np.argwhere(np.diff(np.unique(vec)) <= threshold) + 1)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]
image = get_screencap(device)
print('Got screencap')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(
    grayscale,
    cv2.HOUGH_GRADIENT,
    1,
    minDist=100,
    minRadius=50,
    maxRadius=110,
    param1=90,
    param2=20)

assert circles is not None
circles = np.round(circles[0, :]).astype('int')
circles = circles[(circles[:, 1] > 400) & (circles[:, 1] < 1500)]
assert len(circles) in [n**2 for n in range(4, 10)]
level = int(np.sqrt(len(circles)))
print('level: {}'.format(level))

col_pos = get_almost_unique(circles[:, 0])
row_pos = get_almost_unique(circles[:, 1])
print(col_pos, row_pos)
assert len(col_pos) == level
assert len(row_pos) == level

board = np.zeros((level, level), dtype=np.int8)

for y, x, r in circles:
    myrow = (np.abs(row_pos - x)).argmin()
    mycol = (np.abs(col_pos - y)).argmin()
    assert board[myrow, mycol] == 0
    box = image[x-r:x+r, y-r:y+r]
    box_mean = np.mean(box)
    box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    _, box = cv2.threshold(box, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    box = cv2.resize(box, (80, 80))[15:-15, 15:-15]
    if box_mean < 160:
        # Ostacolo
        cv2.circle(image, (int(y), int(x)), int(r), (0, 255, 255), 10)
        board[myrow, mycol] = -2
    elif box_mean < 180:
        # Numero
        num = 1+np.argmax(np.linalg.norm(box-numbers_templates,
                                         ord='fro', axis=(1, 2)))
        cv2.putText(image, str(num), (y, x),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, 2)
        board[myrow, mycol] = num
    else:
        # Vuoto
        cv2.circle(image, (int(y), int(x)), int(r), (0, 255, 0), 10)

print(board)


def get_counter(x, y):
    north = np.flip(board[:x, y])
    south = board[x+1:, y]
    east = board[x, y+1:]
    west = np.flip(board[x, :y])

    if -2 in north:
        north = north[:np.where(north == -2)[0][0]]
    if -2 in south:
        south = south[:np.where(south == -2)[0][0]]
    if -2 in east:
        east = east[:np.where(east == -2)[0][0]]
    if -2 in west:
        west = west[:np.where(west == -2)[0][0]]

    visible = np.concatenate([north,south,east,west],axis=0)
    number_visible = np.count_nonzero(visible)
    return number_visible, 0 in visible

print(get_counter(4, 3))


#image = cv2.resize(image, None, fx=0.3, fy=0.3)
#cv2.imshow('board', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
