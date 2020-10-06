import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient
import pickle
import os

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blob_blocks_mask = np.asarray(255*(grayimage == 0), dtype=np.uint8)
blob_blocks_mask[:300] = 0
blob_blocks_mask[-450:] = 0
_, blob_blocks_mask, _, _ = cv2.floodFill(
    255-blob_blocks_mask, None, (0, 0), 0)
num, cc = cv2.connectedComponents(blob_blocks_mask)
print(f'Ho trovato {num-1} blocchi')
print(np.unique(cc))
image_cv = np.asarray(cc*255/np.max(cc,axis=(0,1)),dtype=np.uint8)


number_mask = np.asarray(255*(grayimage == 128), dtype=np.uint8)
number_mask[:300] = 0
number_mask[-450:] = 0
ker = np.ones((2, 2))
number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_OPEN, ker)
cnts, _ = cv2.findContours(number_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

with open('digit_classifier_Neural Net.pickle','rb') as f:
    digit_classifier = pickle.load(f)

for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    box = number_mask[y:y+h,x:x+w]
    box = cv2.resize(box,(25,25)).flatten()
    num = digit_classifier.predict([box])[0]
    cv2.putText(image_cv,str(num),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,3,0,3)

cv2.imshow('cv',image_cv)
cv2.imshow('orig',image)
cv2.waitKey()
