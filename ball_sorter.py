import numpy as np
import cv2
from ppadb.client import Client as AdbClient
from sklearn.cluster import KMeans
from math import floor,ceil
from time import sleep

def tuplizza(b):
        return tuple([tuple(x) for x in b.tolist()])

class Board():
    def __init__(self,board):
        self.board = board.copy()
        _, self.N = self.board.shape
        self.visited_board = set()
        self.solution = list()
    
    def is_valid_move(self,num_start_column,num_end_column):
        if num_start_column == num_end_column:
            return False
        start_column = self.board[:,num_start_column]
        end_column = self.board[:,num_end_column]
        if np.all(start_column == 0):
            return False #Nothing to move
        if end_column[0] != 0:
            return False #No space to move
        if np.all(end_column == 0):
            return True #Move to empty column
        start_color = start_column[start_column != 0][0]
        end_color = end_column[end_column != 0][0]
        if start_color != end_color:
            return False #Must be same color
        return True
    
    def move(self,num_start_column,num_end_column):
        assert self.is_valid_move(num_start_column,num_end_column)
        start_pos = np.nonzero(self.board[:,num_start_column])[0][0]
        if np.all(self.board[:,num_end_column] == 0):
            self.board[-1,num_end_column] = self.board[start_pos,num_start_column]
            self.board[start_pos,num_start_column] = 0
        else:
            end_pos = (np.nonzero(self.board[:,num_end_column]))[0][0]-1
            assert self.board[end_pos,num_end_column] == 0
            assert end_pos >= 0
            self.board[end_pos,num_end_column] = self.board[start_pos,num_start_column]
            self.board[start_pos,num_start_column] = 0
    
    def is_win(self):
        return np.unique(self.board, axis=0).shape == (1,self.N)

    def backtrack(self):
        backup = self.board.copy()
        for num_start_column in range(self.N):
            #Se la colonna è già ordinata lasciala stare
            if np.unique(self.board[:,num_start_column]).size == 1:
                continue
            for num_end_column in range(self.N):
                #Se sposto una colonna quasi completa in una colonna vuota lascio stare
                col_from = self.board[:,num_start_column]
                if np.all(self.board[:,num_end_column] == 0) and np.unique(col_from[col_from != 0]).size == 1:
                    continue
                if self.is_valid_move(num_start_column,num_end_column):
                    self.move(num_start_column,num_end_column)
                    if tuplizza(self.board) not in self.visited_board:
                        self.visited_board.add(tuplizza(self.board))
                        if self.backtrack():
                            self.solution.insert(0,(num_start_column,num_end_column))
                            return True
                    if self.is_win():
                        self.solution.append((num_start_column,num_end_column))
                        return True
                    self.board = backup.copy()
        return False
    
    def solve(self):
        self.backtrack()
        shorter_sol = list()
        i = 0
        while i < len(self.solution)-1:
            first_move = self.solution[i]
            second_move = self.solution[i+1]
            if first_move[1] == second_move[0]:
                shorter_sol.append((first_move[0],second_move[1]))
                i += 2
            else:
                shorter_sol.append(first_move)
                i += 1
        shorter_sol.append(self.solution[-1])
        self.solution = shorter_sol


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

print(board)
#print(board[:,:int(ceil(N/2))])
#print(board[:,int(ceil(N/2)):])

assert np.all(board[:,-2:] == 0)
assert all(list(np.ravel(board[:,:-3])).count(i) for i in range(1,N-1))

b = Board(board)
b.solve()

print(b.is_win())
print(b.board)

space = colum_num_to_screen[1][0]-colum_num_to_screen[0][0]
print(space)
colum_num_to_screen[N-2] = (colum_num_to_screen[N-3][0]+space,colum_num_to_screen[N-3][1])
colum_num_to_screen[N-1] = (colum_num_to_screen[N-3][0]+2*space,colum_num_to_screen[N-3][1])

for i,move in enumerate(b.solution):
    print('Move {}: #{}/{}'.format(move,i+1,len(b.solution)))
    xfrom,yfrom = colum_num_to_screen[move[0]]
    xto, yto = colum_num_to_screen[move[1]]
    device.input_tap(xfrom/0.2,yfrom/0.2+150)
    device.input_tap(xto/0.2,yto/0.2+150)
sleep(2)
device.input_tap(530,1340)