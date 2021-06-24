import matplotlib.pyplot as plt
import HopliteSolver as hs
import numpy as np
from cv2 import cv2

solver = hs.Solver()
hs.ioManager.showImg(solver.image)
#plt.show()

image = hs.ioManager.getScreenshot()
h,w,d = image.shape
simplified = np.zeros((h,w,d), dtype=np.uint8)
matrix = np.zeros((h,w,d))
for x in range(-4,4+1,1):
    for y in range(0,10+1,1):
        if x+y>=0 and x+y<=10:
            pos = hs.ioManager.cartesian2pixel((x,y))
            circle_mask = np.zeros((h,w),dtype=np.uint8)
            cv2.circle(circle_mask,pos,30,1,-1)
            mean = cv2.mean(image,mask=circle_mask)
            matrix[x+4,y,:] = np.array(mean[0:3])
            cv2.circle(simplified,pos,50,mean,-1)
            if np.all(np.isclose(np.array(mean),np.array([24,27,65,0]),rtol=0.05)):
                cv2.circle(image,pos,50,(0,0,255),-1)

colors = set()
for x in range(-4,4+1,1):
    for y in range(0,10+1,1):
        colors.add(tuple(matrix[x+4,y,:]))
        #print(matrix[x+4,y,:],end=",")
#print("colors:",len(colors))

plt.subplot(1,2,1)
hs.ioManager.showImg(image)
plt.subplot(1,2,2)
hs.ioManager.showImg(simplified)
#plt.show()

# mydict = {
#     'arcere.png': [(90, 225, 150)],
#     'guerriero.png': [(90, 150, 230)],
#     'mago.png': [(150, 90, 225)],
#     'bombarolo_armato.png': [(90, 90, 230)],
#     'bombarolo.png': [(50, 50, 120)],
#     'bomba_del_bombarolo.png': [(255, 0, 0)],
#     'tempio.png' : [(255,255,255)],
#     'scala.png' : [(200,200,200)]
# }