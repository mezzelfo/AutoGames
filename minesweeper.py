import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
from time import sleep
from random import randint
import math
import itertools
import matplotlib.pyplot as plt
import scipy


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def color_to_digit(hsvpixel):
    assert hsvpixel.shape == (3,)
    if (int(hsvpixel[0])+int(hsvpixel[1])) < 5:
        if hsvpixel[-1] > 188:
            return -1
        else:
            return 0
    else:
        hue = hsvpixel[0]
        if 0 <= hue <= 30:
            return 3
        elif 30 < hue <= 70:
            return 2
        elif 70 < hue <= 100:
            return 6
        elif 100 < hue:
            if hsvpixel[2] > 184:
                return 1
            else:
                return 4
    print(f'Errore: colore {hsvpixel} non riconosciuto')
    print(board)
    plt.matshow(board)
    plt.matshow(board == hsvpixel)
    plt.show()
    exit()


def get_board(image, top_start=455, left_start=28, side=(128+1)*8):
    board = image[top_start:top_start+side, left_start:left_start+side]
    board = cv2.resize(board, (8, 8), interpolation=cv2.INTER_AREA)
    board = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    #print(board[3,0])
    #print(board[5,6])
    #exit()
    board = np.vectorize(color_to_digit, signature='(n)->()')(board)
    return board


def tap(move, device, top_start=455, left_start=28, unit_side=129, long_tap=False):
    x = left_start+move[1]*(unit_side)+unit_side/2
    y = top_start+move[0]*(unit_side)+unit_side/2
    if long_tap:
        device.input_swipe(x, y, x+5, y+5, 300)
    else:
        device.input_tap(x, y)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

tap((0,0),device)

def f(mat):
    tmp = cv2.filter2D(mat, -1, ker, borderType=cv2.BORDER_CONSTANT)
    tmp = cv2.filter2D((tmp == board) * infos, -1, ker, borderType=cv2.BORDER_CONSTANT)
    tmp = np.array((tmp*unknowns) > 0, dtype=np.float32)
    return tmp

while True:
    image = get_screencap(device)
    board = get_board(image)

    ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    unknowns = np.array(board == -1, dtype=np.float32)
    infos = np.array(board >= 1, dtype=np.float32)

    detected_bomb = f(unknowns)
    safe_places = f(detected_bomb) - detected_bomb
    
    if np.all(safe_places == 0):
        print('Nessun posto sicuro')
        break
    for safe_move in zip(*np.where(safe_places > 0)):
        tap(safe_move, device)
    
if np.sum(board == -1) > 10:
    for bomb_pos in zip(*np.where(detected_bomb > 0)):
        tap(bomb_pos, device, long_tap=True)

    fig, axes = plt.subplots(2, 2)
    for ax, mat, name in zip(axes.ravel(),
                            [board, unknowns, detected_bomb, safe_places],
                            ['board','unknowns', 'detected_bomb', 'safe_places']):
        ax.set_title(name)
        ax.matshow(mat)
    plt.show()


#frontier = np.where(cv2.filter2D(infos, -1, ker, borderType=cv2.BORDER_CONSTANT)*unknowns > 0, 1.0, 0.0)
#weights = np.reciprocal(num_near_unknowns, where=num_near_unknowns > 0)
#to_choose = cv2.filter2D(weights, -1, ker, borderType=cv2.BORDER_CONSTANT)*frontier
