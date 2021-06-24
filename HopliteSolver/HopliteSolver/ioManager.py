import subprocess
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

def convertImage(image,format_from='BGR',format_to='GRAY'):
    if (format_from,format_to) == ('BGR','GRAY'):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (format_from,format_to) == ('BGR','RGB'):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError('Non ho trovato la conversione richiesta')

def showImg(imageBGR):
    imageRGB = convertImage(imageBGR,format_from='BGR',format_to='RGB')
    plt.imshow(imageRGB)

def getScreenshot():
    pipe = subprocess.Popen("./adb exec-out screencap -p", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read()
    imageBGR = cv2.imdecode(np.frombuffer(
        image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return imageBGR

def roundToInt(t):
    return tuple([int(round(x)) for x in t])

def pixel2cartesian(c):
    return roundToInt((-(539/117) + c[0]/117, 3373/234 - c[0]/234 - c[1]/126))

def cartesian2pixel(c):
    return roundToInt((539 + 117*c[0], 1526 - 63*c[0] - 126*c[1]))


token_names = [
    'arcere',
    'guerriero',
    'mago',
    'bombarolo_armato',
    'bombarolo',
    'bomba_del_bombarolo',
    'tempio',
    'scala'
]


def detectEntities(imageGRAY):
    entities_positions = dict()
    for tocken_name in token_names:
        template = cv2.imread('HopliteSolver/tokens/'+tocken_name+'.png', 0)
        res = cv2.matchTemplate(imageGRAY, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(res >= threshold)
        points = list(zip(*loc[::-1]))
        #print('Ho trovato',len(points),tocken_name)
        for pt in points:
            cartesian = pixel2cartesian(pt)
            entities_positions[cartesian] = tocken_name
    return entities_positions
           
def getDelta(pos, dx=-30, dy=30):
    return (pos[0]+dx, pos[1]+dy)

def getLava(imageBGR):
    lavapos = set()
    h,w,_ = imageBGR.shape
    for x in range(-4,4+1,1):
        for y in range(0,10+1,1):
            if x+y>=0 and x+y<=10:
                circle_mask = np.zeros((h,w),dtype=np.uint8)
                cv2.circle(circle_mask,cartesian2pixel((x,y)),30,1,-1)
                mean = cv2.mean(imageBGR,mask=circle_mask)
                if np.all(np.isclose(np.array(mean),np.array([24,27,65,0]),rtol=0.05)):
                    lavapos.add((x,y))
    return lavapos