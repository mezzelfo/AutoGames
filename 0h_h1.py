import numpy as np
from cv2 import cv2
from ppadb.client import Client
from itertools import product
from time import sleep

N = 12

def validSeq(seq):
    assert len(seq) == N
    if seq.count(1) != seq.count(-1):
        return False
    for s in range(N):
        if seq[s:s+3] == (-1,-1,-1):
            return False
        if seq[s:s+3] == (1,1,1):
            return False
    return True

def solve_onestep(board):
    nonzerousedseq = board[np.all(board != 0, axis=1)]
    if nonzerousedseq.shape != (0,N):
        myvalid = valid[np.invert((valid == nonzerousedseq[:,None]).all(axis=2).any(axis=0))]
        assert myvalid.shape[0] == valid.shape[0] - nonzerousedseq.shape[0]
    else:
        myvalid = valid
    check = np.all(myvalid*board[:,None]>=0,axis=2) #TODO
    for i in range(N):
        if np.all(board[i,:] != 0):
            continue
        compatibles = myvalid[check[i]]
        m = len(compatibles)
        assert(m > 0)
        fixed = np.sum(compatibles,axis=0)
        fixed[np.abs(fixed) != m] = 0
        fixed = np.divide(fixed,m)
        board[i,:] = fixed
    return board

def solve(board):
    board = solve_onestep(board)#ROW
    board = solve_onestep(board.T).T#ROW
    return board


client = Client()
device = client.devices()[0]

for testNum in range(100):
    image = cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8),cv2.IMREAD_COLOR)
    #cropped = image[425:425+1010,22:-22] EMULATOR
    cropped = image[475:1511,22:-22]
    resized = cv2.resize(cropped,(N,N), interpolation=cv2.INTER_AREA)
    board = np.zeros((N,N),dtype=np.int8)
    board[resized[:,:,1] > 100] = 1 #Blue
    board[resized[:,:,-1] > 100] = -1 #Red
    print(board)
    valid = np.array(list(filter(validSeq,product([-1,1],repeat=N))))
    #print(valid.shape)
    newboard = board.copy()
    counter = 0
    while np.any(newboard == 0):
        newboard = solve(newboard)
        counter += 1
        if counter > 200:
            print(testNum,"Non RISOLTO")
            print(board)
            print(newboard)
            exit(-1)
    
    assert np.all(newboard[board != 0] == board[board != 0])
    print(testNum,"risolto con {} passaggi".format(counter))

    delta = newboard - board
    h,w,_ = cropped.shape
    print(h,w)
    for dx in range(N):
        for dy in range(N):
            x = 22+w/N*(0.5+dx)
            y = 475+h/N*(0.5+dy)#425+h/N*(0.5+dy)
            assert 0 <= x <= 1080
            assert 0 <= y <= 1920
            if delta[dy,dx] == 1:
                device.input_tap(x,y)
                device.input_tap(x,y)
            elif delta[dy,dx] == -1:
                device.input_tap(x,y)



    #device.input_tap(182,1697)    #Exit the game
    sleep(8)
    #device.input_tap(350,850)    #New 4x4 game
    #device.input_tap(550,850)    #New 6x6 game
    #device.input_tap(750,850)    #New 8x8 game
    #device.input_tap(350,1050)   #New 10x10 game
    device.input_tap(550,1050)   #New 12x12 game    
    sleep(1)
