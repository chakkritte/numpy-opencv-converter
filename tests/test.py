import np_opencv_module as npcv
import cv2
import numpy as np

def face_detection_cuda(img, useGPU):
	faces = npcv.face_detection(img,'haarcascade_frontalface_alt2.xml','haarcascade_frontalface_alt2_cuda.xml',useGPU)
	for face in faces:
		x,y,w,h = face
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	return img


img = cv2.imread('3.jpg',0)

cv2.imshow('faces', face_detection_cuda(img,True))
cv2.waitKey(0)
