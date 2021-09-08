import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import constraint
from itertools import accumulate
import operator

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

problem = constraint.Problem()
problem.addVariables([f'r{x}c{y}' for x in range(level) for y in range(level)], [0,1])

for x,y in zip(*np.where(board == -2)):
    problem.addConstraint(constraint.ExactSumConstraint(0), [f'r{x}c{y}'])


def check_counter(X,Y,vars):
    assert 0 <= X < level
    assert 0 <= Y < level
    assert all(v == 0 or v == 1 for v in vars)
    assert len(vars) == 2*(level-1)
    north_len = X
    south_len = level-1-X
    east_len = level-1-Y
    west_len = Y

    north = vars[:north_len]
    south = vars[north_len:north_len+south_len]
    east = vars[north_len+south_len:north_len+south_len+east_len]
    west = vars[north_len+south_len+east_len:]

    
    assert len(north) == north_len
    assert len(south) == south_len
    assert len(east) == east_len
    assert len(west) == west_len

    assert board[X,Y] > 0
    target = sum(sum(accumulate(s, operator.mul)) for s in [north, south, east, west])
    return board[X,Y] == target

def check_counter_generator(A,B):
    return lambda *vs: check_counter(A,B,vs)

for x,y in zip(*np.where(board > 0)):
    target_num = board[x,y]
    problem.addConstraint(constraint.ExactSumConstraint(1), [f'r{x}c{y}'])
    north = [f'r{z}c{y}' for z in range(x-1,-1,-1)]
    south = [f'r{z}c{y}' for z in range(x+1,level)]
    east = [f'r{x}c{z}' for z in range(y+1,level)]
    west = [f'r{x}c{z}' for z in range(y-1,-1,-1)]
    problem.addConstraint(check_counter_generator(x,y), north+south+east+west)
    problem.addConstraint(constraint.MinSumConstraint(target_num), north+south+east+west)

for s in problem.getSolutionIter():
    t = [s[f'r{x}c{y}'] for x in range(level) for y in range(level)]
    t = np.asarray(t).reshape(level,level)
    print(t)
    print()

print(board)
# print(get_counter(4, 3))


#image = cv2.resize(image, None, fx=0.3, fy=0.3)
#cv2.imshow('board', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
