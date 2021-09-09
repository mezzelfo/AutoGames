import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import constraint
from itertools import accumulate, takewhile, product
import operator
from scipy.signal import convolve
from time import sleep

new_level_button = {'4x4':(375, 880),  '5x5':(536, 880), '6x6':(734, 880),
                    '7x7':(375, 1090), '8x8':(536, 1090), '9x9':(734, 1090)}

numbers_templates = np.load('oh_no_numbers.npy')


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


def get_almost_unique(vec, threshold=20):
    return np.delete(np.unique(vec), np.argwhere(np.diff(np.unique(vec)) <= threshold) + 1)

def check_counter(X,Y,lengths,vars):
        l1,l2,l3,_ = lengths

        north = vars[:l1]
        south = vars[l1:l1+l2]
        east = vars[l1+l2:l1+l2+l3]
        west = vars[l1+l2+l3:]

        assert board[X,Y] > 0
        target = sum(sum(accumulate(s, operator.mul)) for s in [north, south, east, west])
        return board[X,Y] == target

def check_counter_generator(x,y,vs_by_dir):
    # get the length of vars
    lengths = [len(v) for v in vs_by_dir]
    assert sum(lengths) <= (board[x,y]+1)*4
    return lambda *vs: check_counter(x,y,lengths,vs)

def get_interesting_variables(r, lambda_take_while, lambda_map, target_num):
    x = takewhile(lambda_take_while, r) #take until first -2 in board in that direction
    x = map(lambda_map, x) # transform to variables name
    return  list(x)[:target_num+1] # get at most target_num+1 in each direction


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

col_pos = sorted(get_almost_unique(circles[:, 0]))
row_pos = sorted(get_almost_unique(circles[:, 1]))
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
    

for x,y in zip(*np.where(board > 0)):
    target_num = board[x,y]
    problem.addConstraint(constraint.ExactSumConstraint(1), [f'r{x}c{y}'])
    north = get_interesting_variables(
        range(x-1,-1,-1), lambda z: board[z,y] >= 0, lambda z: f'r{z}c{y}', target_num
    )

    south = get_interesting_variables(
        range(x+1,level), lambda z: board[z,y] >= 0, lambda z: f'r{z}c{y}', target_num
    )

    east = get_interesting_variables(
        range(y+1,level), lambda z: board[x,z] >= 0, lambda z: f'r{x}c{z}', target_num
    )

    west = get_interesting_variables(
        range(y-1,-1,-1), lambda z: board[x,z] >= 0, lambda z: f'r{x}c{z}', target_num
    )

    vs_by_dir = [north, south, east, west]
    merged_vars = north+south+east+west
    constr = check_counter_generator(x,y,vs_by_dir)
    problem.addConstraint(constr, merged_vars)
    problem.addConstraint(constraint.MinSumConstraint(target_num), merged_vars)

    # check for all possible combinations
    feasibles = np.asarray(list(filter(lambda t: constr(*t),product([0,1], repeat=len(merged_vars)))))
    if len(feasibles) > 0:
        for n,c in enumerate(np.sum(feasibles, axis=0).tolist()):
            vname = merged_vars[n]
            if c == len(feasibles):
                problem.addConstraint(constraint.ExactSumConstraint(1), [vname])
                print('yep', len(merged_vars))
            elif c == 0:
                problem.addConstraint(constraint.ExactSumConstraint(0), [vname])
                print('yop', len(merged_vars))

print('start searching')
sol = problem.getSolution()
print('found one')
sol = [sol[f'r{x}c{y}'] for x in range(level) for y in range(level)]
sol = np.asarray(sol).reshape(level,level)
#fix lonely blue dots
lonely = convolve(sol,[[0,1,0],[1,0,1],[0,1,0]], mode='same', method='direct') == 0

for x in range(level):
    for y in range(level):
        xscreen = col_pos[y]
        yscreen = row_pos[x]
        if board[x,y] == 0:
            device.input_tap(xscreen,yscreen)
            if sol[x,y] == 0 or lonely[x,y]:
                device.input_tap(xscreen,yscreen)

sleep(6)
device.input_tap(*new_level_button['8x8'])
sleep(1)
