from sre_constants import SUCCESS
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread("car image1.jpg")

while True:
    SUCCESS, img = cap.read()
    imgOut = segmentor.removeBG(img, imgBg , threshold=0.70)


    #imgOut = cvzone.stackImages([img,imgOut],2,1)
    _, imgOut = fpsReader.update(imgOut , color=(0,0,255))
    
    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
