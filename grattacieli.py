import numpy as np
from cv2 import cv2
from numpy.lib.npyio import recfromtxt
from ppadb.client import Client as AdbClient
import constraint
import pickle
from itertools import accumulate

def get_screencap(device):
   return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)

client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]
image = get_screencap(device)
_, thresh_img = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
X = 120
Y = 640
N = 4
L = 850
thresh_img[:300,:] = 0
thresh_img[-300:,:] = 0
thresh_img[Y:Y+L,X:X+L] = 0

problem = constraint.Problem()
problem.addVariables([f'r{x}c{y}' for x in range(N) for y in range(N)], list(range(1,N+1)))

for j in range(N):
    v = [f'r{i}c{j}' for i in range(N)]
    problem.addConstraint(constraint.AllDifferentConstraint(),v)
    v = [f'r{j}c{i}' for i in range(N)]
    problem.addConstraint(constraint.AllDifferentConstraint(),v)

with open('digit_classifier_Neural Net.pickle','rb') as f:
    digit_classifier = pickle.load(f)

contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for n,c in enumerate(contours):
    x,y,w,h = cv2.boundingRect(c)
    box = thresh_img[y:y+h,x:x+w]
    box = cv2.resize(box,(25,25)).flatten()
    num = digit_classifier.predict([box])[0]
    tab_pos = ((y-Y)//(L//N),(x-X)//(L//N))
    if tab_pos[0] == -1: #riga in alto
        v = [f'r{i}c{tab_pos[1]}' for i in range(N)]
    elif tab_pos[0] == N: #riga in basso
        v = [f'r{i}c{tab_pos[1]}' for i in range(N-1,-1,-1)]
    elif tab_pos[1] == -1: #colonna a SX
        v = [f'r{tab_pos[0]}c{i}' for i in range(N)]
    elif tab_pos[1] == N: #colonna a DX
        v = [f'r{tab_pos[0]}c{i}' for i in range(N-1,-1,-1)]
    else:
        print('Errore',num,'at',tab_pos,(x,y))
    problem.addVariable(f'num{n}',[num])
    problem.addConstraint(constraint.FunctionConstraint(
            lambda *vals: len(set(accumulate(vals[:-1], max))) == vals[-1]
            ), v+[f'num{n}'])
solutions = problem.getSolutions()
print(f'Ho trovato {len(solutions)} soluzioni')
S = solutions[0]
for r in range(N):
    for c in range(N):
        print(S[f'r{r}c{c}'],end=' ')
    print()