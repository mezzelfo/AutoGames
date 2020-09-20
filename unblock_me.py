import numpy as np
from cv2 import cv2
from ppadb.client import Client as AdbClient

def get_screencap(device):
    return cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)

client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = get_screencap(device)
imagehsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

brown_pieces_mask = cv2.inRange(imagehsv,(10,250,150),(180,255,255))
red_piece_mask = cv2.inRange(imagehsv,(0,250,150),(10,255,255))
cnts, _ = cv2.findContours(brown_pieces_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
brown_pieces_boxes = [cv2.boundingRect(cnt) for cnt in cnts if cv2.contourArea(cnt) > 300]
for x,y,w,h in brown_pieces_boxes:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),5)
cnts, _ = cv2.findContours(red_piece_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
red_piece_box = cv2.boundingRect(max(cnts,key=cv2.contourArea))
x,y,w,h = red_piece_box
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
print(brown_pieces_boxes)
print(red_piece_box)

cv2.imshow('original',image)
cv2.waitKey()