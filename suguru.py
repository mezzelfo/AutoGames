import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import pickle
import os
import constraint

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)

def get_almost_unique(vec, threshold=20):
    return np.delete(np.unique(vec), np.argwhere(np.diff(np.unique(vec)) <= threshold) + 1)

client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayimage[:300] = 200
grayimage[-450:] = 200
edge = cv2.Canny(grayimage,50,150,apertureSize = 3)
number_mask = np.asarray(255*(grayimage == 128), dtype=np.uint8)
ker = np.ones((3, 3))
number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_OPEN, ker)
ker = np.ones((10,10))
dilated_number_mask = cv2.dilate(number_mask,ker)
edge[dilated_number_mask > 0] = 0
verlines = get_almost_unique(np.asarray(np.mean(edge,axis=0) > 5).nonzero())[:-1]
horlines = get_almost_unique(np.asarray(np.mean(edge,axis=1) > 5).nonzero())[:-1]
blob_blocks_mask = np.asarray(255*(grayimage >= 210), dtype=np.uint8)
blob_blocks_mask = blob_blocks_mask | dilated_number_mask
ker = np.ones((4,4))
blob_blocks_mask = cv2.morphologyEx(blob_blocks_mask, cv2.MORPH_CLOSE, ker)
groups_num, cc = cv2.connectedComponents(blob_blocks_mask)
side_len = round(np.mean(np.diff(verlines)))
with open('digit_classifier_Neural Net.pickle','rb') as f:
    digit_classifier = pickle.load(f)
board = np.zeros((len(horlines),len(verlines)),dtype=np.int)
numbers = np.zeros((len(horlines),len(verlines)),dtype=np.int)
variables = np.reshape(np.arange(len(horlines)*len(verlines),dtype=np.int),board.shape)
for i,y in enumerate(horlines):
    for j,x in enumerate(verlines):
        board[i,j] = cc[y+30,x+30]
        cell = number_mask[y:y+side_len,x:x+side_len]
        if np.sum(cell,axis=None) > 10000:
            cnts, _ = cv2.findContours(number_mask[y:y+side_len,x:x+side_len],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            x1,y1,w,h = cv2.boundingRect(cnts[0])
            box = number_mask[y+y1:y+y1+h,x+x1:x+x1+w]
            box = cv2.resize(box,(25,25)).flatten()
            num = digit_classifier.predict([box])[0]
            numbers[i,j] = num
print(board)
print(numbers)
print(variables)

problemsize = np.max(np.bincount(board.ravel()))
print(problemsize)

problem = constraint.Problem()
for group in range(1,groups_num):
    varlist = variables[board == group].tolist()
    groupsize = np.count_nonzero(board == group)
    problem.addVariables(varlist,list(range(1,groupsize+1)))
    problem.addConstraint(constraint.AllDifferentConstraint(),varlist)

for i in range(len(horlines)):
    for j in range(len(verlines)):
        if numbers[i,j] != 0:
            problem.addConstraint(constraint.InSetConstraint([numbers[i,j]]),[variables[i,j]])

padvars = np.pad(variables,1,constant_values = -1)
for i in range(1,len(horlines)+1):
    for j in range(1,len(verlines)+1):
        nearvars = padvars[i-1:i+1,j-1:j+1].ravel()
        nearvars = nearvars[nearvars >= 0].tolist()
        myvar = padvars[i,j]
        #print(myvar,nearvars)
        problem.addConstraint(constraint.AllDifferentConstraint(),nearvars)

solutions = problem.getSolutions()
solution = np.reshape([solutions[0][i] for i in range(len(horlines)*len(verlines))],board.shape)

print('Ho trovato',len(solutions),'soluzioni')

for i,y in enumerate(horlines):
    for j,x in enumerate(verlines):
        if numbers[i,j] == 0:
            device.input_tap(x+side_len//2,y+side_len//2)
            device.input_tap(solution[i,j]*(1080//(problemsize+1)),1600)

cv2.imshow('edge',edge)
cv2.imshow('number_mask',number_mask)
cv2.imshow('blob_blocks_mask',blob_blocks_mask)
cv2.waitKey()
#blob_blocks_mask = np.asarray(255*(grayimage == 0), dtype=np.uint8)
#blob_blocks_mask[:300] = 0
#blob_blocks_mask[-450:] = 0
#_, filled_blob_blocks_mask, _, _ = cv2.floodFill(
#     255-blob_blocks_mask, None, (0, 0), 0)
# num, cc = cv2.connectedComponents(filled_blob_blocks_mask)
# print(f'Ho trovato {num-1} regioni')
# print(np.unique(cc))
# image_cv = np.asarray(cc*255/np.max(cc,axis=(0,1)),dtype=np.uint8)
# a,b,c,d = cv2.boundingRect(np.array(image_cv>0,dtype=np.uint8))
# #cv2.rectangle(image,(a,b),(a+c,b+d),(0,0,255),3)


# number_mask = np.asarray(255*(grayimage == 128), dtype=np.uint8)
# number_mask[:300] = 0
# number_mask[-450:] = 0
# ker = np.ones((2, 2))
# number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_OPEN, ker)
# cnts, _ = cv2.findContours(number_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# with open('digit_classifier_Neural Net.pickle','rb') as f:
#     digit_classifier = pickle.load(f)

# for cnt in cnts:
#     x,y,w,h = cv2.boundingRect(cnt)
#     box = number_mask[y:y+h,x:x+w]
#     box = cv2.resize(box,(25,25)).flatten()
#     num = digit_classifier.predict([box])[0]
#     #cv2.putText(image_cv,str(num),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,3,0,3)

# edges = cv2.Canny(grayimage,50,150,apertureSize = 3)
# edges[:300] = 0
# edges[-450:] = 0
# lines = cv2.HoughLines(edges,1,np.pi/180,300)

# lines = [(l[0][0],l[0][1]) for l in lines]
# lines = [(r*np.cos(t),r*np.sin(t)) for (r,t) in lines]
# horlines = [int(line[1]) for line in lines if abs(line[0]) < 1]
# verlines = [int(line[0]) for line in lines if abs(line[1]) < 1]
# horlines = get_almost_unique(horlines)[:-1]
# verlines = get_almost_unique(verlines)[:-1]

# board = np.zeros((len(verlines),len(horlines)))

# for i,x0 in enumerate(verlines):
#     for j,y0 in enumerate(horlines):
#         board[i,j] = cc[y0+100,x0+100]
#         cv2.circle(image,(x0+100,y0+100),10,(0,255,0),-1)

# print(board.transpose())

# #cv2.imshow('image_cv',image_cv)
# cv2.imshow('image',image)
# #cv2.imshow('a',edges)

# cv2.waitKey()
