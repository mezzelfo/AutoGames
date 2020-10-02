import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import networkx as nx
from queue import Queue


def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)
imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
all_pieces_mask = cv2.inRange(imagehsv, (0, 230, 150), (180, 255, 255))
ker = np.ones((15, 15))
all_pieces_mask = cv2.erode(all_pieces_mask, ker)
num_blocks, cc = cv2.connectedComponents(all_pieces_mask)
starting_board = cc[650:1500:165, 120:970:165]
starting_board = tuple(tuple(l) for l in starting_board)
nums = np.max(starting_board)
tile_infos = {}
for n in range(1, nums+1):
    r, c = np.where(np.asarray(starting_board) == n)
    if r[0] != r[1]:
        tile_infos[n] = {'dir': 'v', 'pos': (r[0], c[0])}
    else:
        tile_infos[n] = {'dir': 'h', 'pos': (r[0], c[0])}
        if r[0] == 2:
            target = n
Q = Queue()
Q.put(starting_board)

G = nx.DiGraph()
G.add_node(starting_board)


def get_newpos(n, pos, move):
    if tile_infos[n]['dir'] == 'v':
        return [(move+d, k[1]) for d, k in enumerate(pos)]
    else:
        return [(k[0], move+d) for d, k in enumerate(pos)]


def get_next_states(board):
    board = np.asarray(board)
    copy = board.copy()

    for n in range(1, nums+1):
        others = list(zip(*np.where((board != n) & (board != 0))))
        pos = list(zip(*np.where(board == n)))
        if pos[0][0] != pos[1][0]:
            imp = pos[0][0]
        else:
            imp = pos[0][1]
        # verso l'alto
        for m in range(imp-1, -1, -1):
            newpos = get_newpos(n, pos, m)
            if newpos[0] in others:
                break
            board[board == n] = 0
            for p in newpos:
                board[p] = n
            yield n, m, tuple(tuple(l) for l in board)
            board = copy.copy()
        # verso il basso
        for m in range(imp+1, 6-len(pos)+1):
            newpos = get_newpos(n, pos, m)
            if newpos[-1] in others:
                break
            board[board == n] = 0
            for p in newpos:
                board[p] = n
            yield n, m, tuple(tuple(l) for l in board)
            board = copy.copy()

# 650:1500:165, 120:970:165


def board2pixel(pos):
    return (120+pos[1]*165, 650+pos[0]*165)


while not Q.empty():
    board = Q.get()
    if board[2][-1] == target:
        print('Vinto')
        winning_path = nx.bidirectional_shortest_path(G, starting_board, board)
        for i in range(len(winning_path)-1):
            # print(np.asarray(winning_path[i]))
            link = G[winning_path[i]][winning_path[i+1]]
            block = link['block']
            actual_pos = tile_infos[block]['pos']
            if tile_infos[block]['dir'] == 'h':
                future_pos = (actual_pos[0], link['move'])
            else:
                future_pos = (link['move'], actual_pos[1])
            startx, starty = board2pixel(actual_pos)
            endx, endy = board2pixel(future_pos)
            print(
                f'Moving {block} from {actual_pos} to {future_pos} -> {(startx,starty)} to {endx,endy}')
            device.input_swipe(startx, starty, endx, endy, 500)
            tile_infos[block]['pos'] = future_pos
        print(len(winning_path)-1)
        break
    for n, move, state in get_next_states(board):
        if state not in G:
            Q.put(state)
            G.add_node(state)
        G.add_edge(board, state, block=n, move=move)
