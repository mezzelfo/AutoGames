import numpy as np
from cv2 import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
from PIL import Image,ImageDraw,ImageFont

from ppadb.client import Client as AdbClient
import os



X = []
y = []
fonts = [
    #ImageFont.truetype("/usr/share/fonts/truetype/malayalam/AnjaliOldLipi-Regular.ttf", 80, encoding="unic"),
    ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 80, encoding="unic"),
    ImageFont.truetype("/usr/share/fonts/truetype/malayalam/Keraleeyam-Regular.ttf", 80, encoding="unic"),
]
for number in range(0,11):
    for font in fonts:
        for thickness in range(4):
            for _ in range(10):
                canvas = Image.new('L', (220,220),0)
                draw = ImageDraw.Draw(canvas)
                draw.text((20,20), str(number), 255, font, stroke_width=thickness)
                canvas = np.array(canvas)
                
                left,top,w,h = cv2.boundingRect(canvas)
                canvas = cv2.resize(canvas[top:top+h,left:left+w],(25,25))
                canvas += cv2.randn(np.zeros((25,25),dtype=np.uint8),0,1)
                
                #cv2.imshow('canvas',canvas)
                #cv2.waitKey()
                X.append(canvas.flatten())
                y.append(number)
X = np.asarray(X)
y = np.asarray(y)
print(X.shape,y.shape)
#exit()

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.001, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name,score)
    if score >= 1.0:
        print(f"Sto salvando {name}")
        with open(f"digit_classifier_{name}.pickle","wb") as f:
            pickle.dump(clf,f)


client = AdbClient(host="127.0.0.1", port=5037)
device = client.devices()[0]

image = cv2.imdecode(np.frombuffer(device.screencap(), dtype=np.uint8), cv2.IMREAD_COLOR)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # 0hn0
# mask = cv2.inRange(image_hsv, (0, 0, 254), (180, 255, 255))
# cv2.floodFill(mask,None,(10,300),0)
# mask[:300] = 0
# mask[-300:] = 0

# # suguru
# mask = cv2.inRange(image_hsv, (0, 0, 126), (255, 255, 128))
# mask[:300] = 0
# mask[-450:] = 0
# ker = np.ones((2, 2))
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)

# tende e alberi
mask = cv2.inRange(image_hsv, (0, 51, 0), (8, 255, 255))
ker = np.ones((3,3))
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,ker)


cnts, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
number_mask_back = mask.copy()
for digit_classifier_file in os.listdir('./'):
    if '.pickle' in digit_classifier_file:
        number_mask = number_mask_back.copy()
        with open('digit_classifier_Linear SVM.pickle','rb') as f:
            digit_classifier = pickle.load(f)

        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            box = number_mask[y:y+h,x:x+w]
            box = cv2.resize(box,(25,25)).flatten()
            num = digit_classifier.predict([box])[0]
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(image,str(num),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),3)
            
        cv2.imshow(digit_classifier_file, image)
cv2.waitKey()