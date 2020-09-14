import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
from time import sleep
from random import randint
import math
import itertools
import matplotlib.pyplot as plt

numbers_templates = np.load('tende_e_alberi_numbers.npy')

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def get_almost_unique(vec, threshold=4):
    return np.delete(np.unique(vec), np.argwhere(np.diff(np.unique(vec)) <= threshold) + 1)


def get_num(box):
    if np.all(box == 0):
        return 0
    kernel = np.ones((5, 5), np.uint8)
    box = cv2.dilate(box, kernel)
    contours = np.asarray(cv2.findContours(
        box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    assert len(contours) == 1
    (x,y,w,h) = cv2.boundingRect(contours[0])
    box = cv2.resize(box[y:y+h,x:x+w],(25,25))
    scores = np.sum(
    (box.astype(float)-numbers_templates.astype(float))**2, axis=(1, 2))
    return np.argmin(scores)


def checkPartial(sol):
    tends = [t for t in sol if t is not None]

    if any([len([t for t in tends if t[0] == i]) > vertical_numbers[i] for i in range(level)]):
        return False

    if any([len([t for t in tends if t[1] == i]) > horizontal_numbers[i] for i in range(level)]):
        return False

    for (t1, t2) in itertools.combinations(tends, 2):
        if abs(t1[0]-t2[0]) <= 1 and abs(t1[1]-t2[1]) <= 1:
            return False
    return True


def backtrack(idx):
    if idx >= len(alberi_pos):
        return solution
    for t in possibilita[idx]:
        solution[idx] = t
        if checkPartial(solution):
            if backtrack(idx+1):
                return solution
    solution[idx] = None
    return False


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

for _ in range(20):
    image = get_screencap(device)
    print('Immagine acquisita')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(grayscale, 50, 255, cv2.THRESH_BINARY_INV)[1]
    binary[:300, :] = 0
    contours = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = filter(lambda cnt: cv2.contourArea(cnt) > 1000, contours)
    boundingRects = map(lambda cnt: cv2.boundingRect(cnt), contours)
    black_centroids = np.asarray(list(map(lambda rect: (
        rect[0]+round((rect[2]+rect[3])/4), rect[1]+round((rect[2]+rect[3])/4)), boundingRects)))

    col_pos = get_almost_unique(black_centroids[:, 0])
    row_pos = get_almost_unique(black_centroids[:, 1])
    assert len(col_pos) == len(row_pos)
    level = len(col_pos)

    alberi_pos = list()
    for i, col in enumerate(col_pos):
        for j, row in enumerate(row_pos):
            if np.min(np.linalg.norm(black_centroids-(col, row), axis=1)) > 20:
                alberi_pos.append((j, i))

    numbers_layer = cv2.inRange(cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV), (5, 45, 120), (10, 220, 150)) #150->220
    r = int(round((np.mean(np.diff(row_pos))+np.mean(np.diff(col_pos)))/2))
    horizontal_numbers = list()
    for (x, y) in ((col-r//2, row_pos[0]-r-r//2) for col in col_pos):
        horizontal_numbers.append(get_num(numbers_layer[y:y+r, x:x+r]))

    vertical_numbers = list()
    for (x, y) in ((col_pos[0]-r-r//2, row-r//2) for row in row_pos):
        if x < 0:
            box = numbers_layer[y:y+r, 0:x+r]
            box = cv2.copyMakeBorder(box, 0, 0, abs(
                x), 0, cv2.BORDER_CONSTANT, value=0)
        else:
            box = numbers_layer[y:y+r, x:x+r]
        assert box.shape == (r, r)
        vertical_numbers.append(get_num(box))

    print('Immagine analizzata')
    print(horizontal_numbers)
    print(vertical_numbers)

    north = [(x-1, y) if x-1 >= 0 and ((x-1, y) not in alberi_pos)
            else None for (x, y) in alberi_pos]
    south = [(x+1, y) if x+1 < level and ((x+1, y) not in alberi_pos)
            else None for (x, y) in alberi_pos]
    west = [(x, y-1) if y-1 >= 0 and ((x, y-1) not in alberi_pos)
            else None for (x, y) in alberi_pos]
    east = [(x, y+1) if y+1 < level and ((x, y+1) not in alberi_pos)
            else None for (x, y) in alberi_pos]
    possibilita = [[p for p in a if p is not None]
                for a in zip(north, south, west, east)]

    print('Ricerca soluzione iniziata')

    solution = list([None for _ in range(len(alberi_pos))])
    sol = backtrack(0)

    print('Soluzione trovata')

    for j, i in sol:
        device.input_tap(col_pos[i], row_pos[j])
        device.input_tap(col_pos[i], row_pos[j])

    print('Soluzione inserita')

    sleep(6)
    device.input_tap(750,1080)
    sleep(4)

# for j, i in alberi_pos:
#    cv2.circle(image, (col_pos[i], row_pos[j]), 20, (255, 0, 0), -1)
# for j, i in sol:
#    cv2.circle(image, (col_pos[i], row_pos[j]), 20, (0, 0, 255), -1)
#cv2.imshow('image', cv2.resize(image, None, fx=0.3, fy=0.3))
# cv2.waitKey()
# cv2.destroyAllWindows()
