import cv2
import numpy as np 
import gui

def implementBrightness(value):
	global editionMade,editedImage,editedImageIndex,flag
	if flag==0:
		editionMade=1
		img=editedImage[editedImageIndex]
		img=imf.Brightness(img,value)
		editedImageIndex+=1
		editedImage+=(img,)
		flag=1
		updateTrackbarPos()
	else:
		img=editedImage[editedImageIndex-1]
		img=imf.Brightness(img,value)
		editedImage[editedImageIndex]=img
	showEditedImage()


