from concurrent.futures import thread
#Creating classes to save the iamge in it and displaying
from pydoc import classname
#For image reading purpose
from sre_constants import SUCCESS
#Importing OpenCV Library for basic image processing functions
import cv2
#calculating looking that image is similar or not 
from cv2 import threshold
# Numpy for array related functions
import numpy as np
import os
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
#creating frame and genrating sound
from pygame import mixer
#to nsert the image in the frame's background
import cvzone

mixer.init()
sound = mixer.Sound('alarm.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0

cap = cv2.VideoCapture(0)
while True:
    SUCCESS, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findId(img2,deslist)
    if id != -1:

        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        face_frame = frame.copy()
        #detected face in faces array
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            #The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36],landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42],landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
            #Now judge what to do for the eye blinks
            if(left_blink==0 or right_blink==0):
                sleep+=1
                drowsy=0
                active=0
                if(sleep>6):
                    status="Plase wake up"
                    color = (255,0,0)

            else:
                drowsy=0
                sleep=0
                active+=1
                if(active>6):
                    status="Active :)"
                    color = (0,255,0)
        
            cv2.putText(imgOriginal,classNames[id],(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.putText(face_frame, status, (150,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
            if(sleep>6):
                try:
                    sound.play()
                except:
                    pass

            for n in range(0, 68):
                (x,y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 0, 0), -1)

            #img_arr = np.hstack((imgOriginal, face_frame))
            #cv2.imshow("input images",img_arr)
            #cv2.imshow("Images", imgOriginal)
            cv2.imshow("Frame", face_frame)
            cv2.waitKey(1)
        

