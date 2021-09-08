import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
from sklearn.cluster import KMeans
from math import floor,ceil
from time import sleep
import heapq
from copy import deepcopy

def heuristic_column(col):
    if col[-4] == 0 or col[-3] == 0:
        return 0
    if col[-2] == 0:
        if col[-1] == col[-2]:
            return 0
        else:
            return 1
    if col[-1] == 0:
        if col[-1] != col[-2]:
            return 2
        if col[-2] != col[-3]:
            return 1
        return 0

    if col[-1] != col[-2]:
        return 3
    if col[-2] != col[-3]:
        return 2
    if col[-3] != col[-4]:
        return 1
    return 0

def heuristic(board):
    x1 = sum(map(heuristic_column, board))
    x2 = sum(sum(i != j for i,j in zip(t[:-1], t[1:])) for t in board)
    x3 = sum(len(set(t))-1 for t in board)
    return x1+x2+x3#min(x1,x2)
    
def is_valid_move(board, num_start_column, num_end_column):
    if num_start_column == num_end_column:
        return False
    start_column = board[num_start_column]
    end_column = board[num_end_column]
    if start_column[-1] == 0 or end_column[0] != 0:
        return False
    if all(c == 0 for c in end_column):
        return True
    start_top = next(c for c in start_column if c != 0)
    end_top = next(c for c in end_column if c != 0)
    if start_top == end_top:
        return True
    else:
        return False

def get_neighbors(board):
    N = len(board)
    for num_start_column in range(N):
        for num_end_column in range(N):
            if is_valid_move(board, num_start_column, num_end_column):
                new = [list(t) for t in board]
                start_pos = [i for i,c in enumerate(board[num_start_column]) if c != 0][0]
                new[num_start_column][start_pos] = 0              

                if board[num_end_column][-1] == 0:
                    new[num_end_column][-1] = board[num_start_column][start_pos]
                else:
                    end_pos = [i for i,c in enumerate(board[num_end_column]) if c == 0][-1]
                    new[num_end_column][end_pos] = board[num_start_column][start_pos]
                
                yield tuple(tuple(t) for t in new) , (num_start_column, num_end_column)

def is_winning(board):
    return all(len(set(b)) == 1 for b in board)

def a_star_search(start):
    heap = []
    heapq.heappush(heap, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while heap:
        current = heapq.heappop(heap)[1]
        
        if is_winning(current):
            break
        
        for next, move in get_neighbors(current):
            new_cost = 1 + cost_so_far[current]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next)
                heapq.heappush(heap, (priority, next))
                came_from[next] = (current,move)
    
    b = current
    list_of_moves = []
    while came_from[b]:
        b,move =  came_from[b]
        list_of_moves.append(move)
    list_of_moves.reverse()
    print(len(cost_so_far),len(list_of_moves))
    return list_of_moves

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)

image = image[150:-150, :]
image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
print(image.shape)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(
    grayscale,
    cv2.HOUGH_GRADIENT,
    1,
    minDist=10,
    minRadius=5,
    maxRadius=10,
    param1=90,
    param2=20)
assert circles is not None

circles_pos = np.round(circles[0, :, :-1]).astype('int')
assert len(circles_pos) % 4 == 0
N = int(len(circles_pos)/4)+2
print('Numero di provette', N)
hues = np.asarray([image[y, x, :] for (x, y) in circles_pos])

kmeans_hues = KMeans(n_clusters=N-2, random_state=0).fit(hues)
col_pos = list(np.unique(circles_pos[:,0]))
row_pos = list(np.unique(circles_pos[:,1]))
#Eliminare elementi vicini in col_pos e row_pos
th = 4
col_pos = np.delete(col_pos, np.argwhere(np.diff(col_pos) <= th) + 1)
row_pos = np.delete(row_pos, np.argwhere(np.diff(row_pos) <= th) + 1)

board = np.zeros((4, N), dtype=np.int8)
colum_num_to_screen = dict()
for (col,row),hue in zip(circles_pos,kmeans_hues.labels_):
    myrow = (np.abs(row_pos - row)).argmin()
    mycol = (np.abs(col_pos - col)).argmin()
    if N % 2 == 1 and N > 5:
        if mycol > N - 5:
            mycol += 2
        print('before',mycol)
        mycol = floor(mycol/2)
        print('after',mycol)
    if myrow > 3:
        mycol += ceil(N/2)
    board[myrow % 4, mycol] = hue+1
    colum_num_to_screen[mycol] = (col,row)

board = tuple([tuple(x) for x in board.T])
solution = a_star_search(board)

space = colum_num_to_screen[1][0]-colum_num_to_screen[0][0]
colum_num_to_screen[N-2] = (colum_num_to_screen[N-3][0]+space,colum_num_to_screen[N-3][1])
colum_num_to_screen[N-1] = (colum_num_to_screen[N-3][0]+2*space,colum_num_to_screen[N-3][1])

for i,move in enumerate(solution):
    print('Move {}: #{}/{}'.format(move,i+1,len(solution)))
    xfrom,yfrom = colum_num_to_screen[move[0]]
    xto, yto = colum_num_to_screen[move[1]]
    device.input_tap(xfrom/0.2,yfrom/0.2+150)
    device.input_tap(xto/0.2,yto/0.2+150)
sleep(2)
device.input_tap(530,1340)