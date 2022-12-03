#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
#creating frame and genrating sound
from pygame import mixer
#to nsert the image in the frame's background
import cvzone
#use for image insert
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from concurrent.futures import thread
from pydoc import classname
from sre_constants import SUCCESS
from sys import flags
from cv2 import threshold
from matplotlib.pyplot import flag
import numpy as np
import os

path = 'Train_Image'
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

mixer.init()
sound1 = mixer.Sound('alarm1.wav')
sound2 = mixer.Sound('alarm2.wav')
#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#now the model is Dlib starts from here
#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #SUCCESS, img = cap.read()
    #imgOut = segmentor.removeBG(img, imgBg , threshold=0.70)

    #imgOut = cvzone.stackImages([img, face_frame],2,1)
    #_, imgOut = fpsReader.update(imgOut , color=(0,0,255))

    faces = detector(gray)
    face_frame = frame.copy()
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):
                status="Boss Wake Up"
                color = (255,0,0)

        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="Active :)"
                color = (0,255,0)

        cv2.putText(face_frame, status, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,2)
        if(sleep>6):
            try:
                sound2.play()
            except:
                pass

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 0, 0), -1)

            #imgOut = cvzone.stackImages([img,face_frame],2,1)
            #_, imgOut = fpsReader.update(imgOut , color=(0,0,255))
    
    #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Frame", 1000, 750)
    img2 = cv2.resize(face_frame, (1000,750))

    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    id = findId(img2,deslist)
    if id != -1:
        try:
            sound1.play()
        except:
            pass
        #sound2.play()
        cv2.rectangle(imgOriginal, (0, 0), (200, 200), (0, 255, 0), 3)
        cv2.putText(imgOriginal,classNames[id],(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow("Images", imgOriginal)
    # cv2.imshow("Frame", face_frame)
    #cv2.imshow("Result of detector", face_frame)
    #cv2.imshow("Image", imgOut)
    #cv2.waitKey(1)
    key = cv2.waitKey(1)
    if key == 27:
          break