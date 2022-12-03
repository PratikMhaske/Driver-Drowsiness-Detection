from concurrent.futures import thread
from pydoc import classname
from sre_constants import SUCCESS
from sys import flags
import cv2
from cv2 import threshold
from matplotlib.pyplot import flag
import numpy as np
import os

path = 'Query_Image'
orb = cv2.ORB_create(nfeatures=1000)

images = []
classNames = []
myList = os.listdir(path)
print(myList)
print('Total classes detected', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images):
    deslist=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist

def findId(img, deslist, thres=15):
    kp,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchlist = []
    finalVal = -1
    try:      
        for des in deslist:
            matches = bf.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchlist.append(len(good))
    except:
        pass
    #print(matchlist)
    if len(matchlist)!=0:
        if max(matchlist) > thres:
            finalVal = matchlist.index(max(matchlist))
    return finalVal

deslist = findDes(images)
print(len(deslist))

cap = cv2.VideoCapture(0)
while True:
    SUCCESS, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    id = findId(img2,deslist)
    if id != -1:
        cv2.putText(imgOriginal,classNames[id],(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow("Images", imgOriginal)
    cv2.waitKey(1)