#USING ORB
import cv2
import numpy as np
import os

img1 = cv2.imread("Query_Image\Old_Monk1.jpg",0)
img2 = cv2.imread("Train_Image\Beer.jpg",0)

orb = cv2.ORB_create(nfeatures=1000)

kp1 , des1 = orb.detectAndCompute(img1,None)
kp2 , des2 = orb.detectAndCompute(img2,None)

# imgkp1 = cv2.drawKeypoints(img1,kp1,None)
# imgkp2 = cv2.drawKeypoints(img2,kp2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good= []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

# imgkp1 = cv2.resize(imgkp1, (385,560))
# cv2.imshow("Kp1",imgkp1)
img1 = cv2.resize(img1, (385,560))
cv2.imshow("Images1",img1)

img2 = cv2.resize(img2, (385,560))
# cv2.imshow("Kp2",imgkp2)
cv2.imshow("Images2",img2)

# img3 = cv2.resize(img3, (385,300))
cv2.imshow("Images3",img3)
cv2.waitKey(0)


















##USING YOLO##
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# from IPython.display import Image, display

# FILE_NAME = "popat.jpg"

# display(Image(FILE_NAME, width = 800, height = 700))

# img = cv2.imread(FILE_NAME) 
# bbox, label, conf = cv.detect_common_objects(img)
# for l, c in zip(label, conf):
#     print(f"Detected object: {l} with confidence level of {c}n")
# output_image = draw_bbox(img, bbox, label, conf)
# while(True):
#     output_image = cv2.resize(output_image, (800,650))
#     cv2.imshow("OUT_FILE_NAME", output_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break 

# import numpy as np
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox

# cap = cv2.VideoCapture(0)

# while(True):
#     Capture frame-by-frame
#     ret, frame = cap.read()

#     Detect objects and draw on screen
#     bbox, label, conf = cv.detect_common_objects(frame)
#     output_image = draw_bbox(frame, bbox, label, conf)

#     cv2.imshow('output',output_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()










