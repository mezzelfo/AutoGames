import subprocess
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from os import path

def showImg(imageHSV):
    imageRGB = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2RGB)
    plt.imshow(imageRGB)

def showMask(mask):
    plt.imshow(mask)

def getScreenshot():
    pipe = subprocess.Popen("./adb exec-out screencap -p", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read()
    imageBGR = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    imageHSV = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2HSV)
    return imageHSV

imageHSV = getScreenshot()
bombaroli = cv2.inRange(imageHSV, (0,50,100), (10,255,255))
arceri = cv2.inRange(imageHSV, (40,50,100), (70,255,255))
maghi = cv2.inRange(imageHSV, (130,50,50), (170,255,255))
energia = cv2.inRange(imageHSV, (20,50,50), (40,255,255))
guerrieri = cv2.inRange(imageHSV, (10,150,30), (20,255,255))
lava = cv2.inRange(imageHSV, (0,150,50), (255,255,100))
ioescala = cv2.inRange(imageHSV, (0,0,200), (255,50,255))


for i,mask in enumerate([bombaroli,arceri,maghi,energia,guerrieri,lava,ioescala]):
    plt.subplot(2,4,i+2)
    showMask(mask)

plt.subplot(2,4,1)
showImg(imageHSV)

plt.show()