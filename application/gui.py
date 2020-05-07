import MenuCreatorAndPointArray as mc 
import cv2
import numpy as np
import pydialog 
import imgfunctions as imf
# from win32api import GetSystemMetrics

# cv2.namedWindow('edition',cv2.WINDOW_NORMAL)

imgOpened=False

#edition global variables
sysWidth,sysHeight=pydialog.getSysDimensions()
sysHeight=int(0.92*sysHeight)
height=0
width=0
maxKernel=0
flag=0
originalImage=None
editedImage=[]
editedImageIndex=0

#menu global variables
traversedMenuList=[0]
currentMenu=0
change=0

#dialog global variables
editionMade=0
Cannyth1=0
Cannyth2=255
currentDialog=0

screenImgHeight=int(sysHeight*0.8)
screenImgWidth=int(sysWidth*0.7)
colorgray=(100,85,89)
colordarkgray=(80,65,69)
interface=np.zeros((sysHeight,sysWidth,3),np.uint8)
interface[:]=colorgray
screenEditionStartCol=int(sysWidth/90)
screenEditionEndCol=screenEditionStartCol+screenImgWidth
imgDiff=(sysHeight-screenImgHeight)//2
screenEditionStartRow=imgDiff
screenEditionEndRow=imgDiff+screenImgHeight
screenMenuStartRow=screenEditionStartRow
screenMenuEndRow=screenEditionEndRow
screenMenuEndCol=sysWidth-int(1.2*screenEditionStartCol)
screenMenuStartCol=screenMenuEndCol-410
xPad=screenMenuStartCol
yPad=screenMenuStartRow
interface[screenEditionStartRow:screenEditionEndRow,screenEditionStartCol:screenEditionEndCol]=colordarkgray
rowMenuEndRow=screenEditionStartRow-int(0.7*screenEditionStartCol)
rowMenuStartRow=rowMenuEndRow-50
interface[rowMenuStartRow:rowMenuEndRow,screenEditionStartCol:screenEditionEndCol]=colordarkgray
yyPad=rowMenuStartRow
xxPad=screenEditionStartCol
compare=0
delIndex=-1
cv2.circle(interface,(sysWidth-396,sysHeight-14), 8, (255,255,255), 1)
cv2.putText(interface,'',(sysWidth-400,sysHeight-11),cv2.FONT_HERSHEY_SIMPLEX,0.47,(255,255,255),1)
cv2.putText(interface,'',(sysWidth-400,sysHeight-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)


def exceptionMenu():
	if(editedImage):
		setCurrentMenu(1)


def screen():
	global interface
	cv2.namedWindow('',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('',sysWidth,sysHeight)
	showScreen()

def showScreen():
	cv2.imshow('',interface)

def screenEdition():
	global screenEditionStartRow,screenEditionEndRow,screenEditionStartCol,screenEditionEndCol
	global interface
	global editedImageIndex
	global editedImage
	global height,width,screenImgHeight,screenImgWidth
	global delIndex

	if(compare):
		Compare()
	else:
		if(delIndex>0):
			img=screenImagePadding(editedImage[len(editedImage)-1],height,width,screenImgHeight,screenImgWidth)
		else:
			img=screenImagePadding(editedImage[editedImageIndex],height,width,screenImgHeight,screenImgWidth)
		interface[screenEditionStartRow:screenEditionEndRow,screenEditionStartCol:screenEditionEndCol]=img
	showScreen()

def screenMenu():
	global menu,currentMenu
	img=menu[currentMenu][0]
	interface[screenMenuStartRow:screenMenuEndRow,screenMenuStartCol:screenMenuEndCol]=img
	interface[rowMenuStartRow:rowMenuEndRow,screenEditionStartCol:screenEditionEndCol]=menu[8][0]
	cv2.imshow('',interface)


def screenImagePadding(img,he,wi,h,w):
	background=np.zeros((h,w,3),np.uint8)
	background[:]=colordarkgray
	if(he/wi > h/w):
		newWidth=int(h*wi/he)
		resized=cv2.resize(img, (newWidth,h), interpolation = cv2.INTER_AREA)
		startCol=((w)-newWidth)//2
		endCol=((w)+newWidth)//2
		background[:,startCol:endCol]=resized
	else:
		newHeight=int((w)*he/wi)
		resized=cv2.resize(img, (w,newHeight), interpolation = cv2.INTER_AREA)
		startRow=(h-newHeight)//2
		endRow=(h+newHeight)//2
		background[startRow:endRow,:]=resized
	return background


def Compare():
	global originalImage
	global editedImage
	global editedImageIndex
	global screenEditionStartRow,screenEditionEndRow,screenEditionStartCol,screenEditionEndCol
	global screenImgHeight
	global screenImgWidth

	# im1=screenImagePadding(originalImage,screenImgWidth//2,screenImgHeight)
	# im2=screenImagePadding(editedImage[editedImageIndex],screenImgWidth//2,screenImgHeight)
	im1=originalImage
	if(delIndex>0):
		im2=editedImage[len(editedImage)-1]
	else:
		im2=editedImage[editedImageIndex]
	img=cv2.hconcat([im1,im2])
	img=screenImagePadding(img,img.shape[0],img.shape[1],screenImgHeight,screenImgWidth)

	if(img.shape[0]==screenImgHeight):
		compStartRow=screenEditionStartRow
		compEndRow=screenEditionEndRow
		compStartCol=screenEditionStartCol+((screenImgWidth-img.shape[1])//2)
		compEndCol=compStartCol+img.shape[1]
	else:
		compStartCol=screenEditionStartCol
		compEndCol=screenEditionEndCol
		compStartRow=screenEditionStartRow+((screenImgHeight-img.shape[0])//2)
		compEndRow=compStartRow+img.shape[0]

	interface[screenEditionStartRow:screenEditionEndRow,screenEditionStartCol:screenEditionEndCol]=colordarkgray
	interface[compStartRow:compEndRow,compStartCol:compEndCol]=img


def toggleCompare():
	global compare
	compare=(compare+1)%2
	screenEdition()


cv2.resizeWindow('edition',sysWidth//2,sysHeight)
# h=cv2.imread(pydialog.getFile())
# cv2.imwrite(pydialog.saveFile().name,h)
#################################################################################################	
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
def falseFlag(img):
	global editionMade,editedImageIndex,editedImage,flag,delIndex
	editionMade=1
	if(editedImageIndex<(len(editedImage)-1)):
		delIndex=editedImageIndex
	editedImageIndex+=1
	editedImage+=(img,)
	flag=1
	updateTrackbarPos()

def trueFlag(img):
		editedImage[len(editedImage)-1]=img

def imagePadding(img):
	global height,width
	global sysHeight,sysWidth
	w=sysWidth//2
	h=sysHeight
	background=np.ones((sysHeight,sysWidth//2,3),np.uint8)*127
	if(height/width > h/w):
		newWidth=int(h*width/height)
		resized=cv2.resize(img, (newWidth,h), interpolation = cv2.INTER_AREA)
		startCol=((w)-newWidth)//2
		endCol=((w)+newWidth)//2
		background[:,startCol:endCol]=resized
	else:
		newHeight=int((w)*height/width)
		resized=cv2.resize(img, (w,newHeight), interpolation = cv2.INTER_AREA)
		startRow=(h-newHeight)//2
		endRow=(h+newHeight)//2
		background[startRow:endRow,:]=resized
	return background



#Menu functions:
	#common tiles:
def goBack():
	global traversedMenuList
	global currentMenu
	global change
	traversedMenuList.pop(len(traversedMenuList)-1)
	currentMenu=traversedMenuList[len(traversedMenuList)-1]
	change=1
	screenMenu()
	# cv2.imshow('Menu',menu[currentMenu][0])

def setCurrentMenu(i):
	global currentMenu
	global menu
	global traversedMenuList
	traversedMenuList+=(i,)
	currentMenu=i
	# cv2.imshow('Menu',menu[currentMenu][0])
	screenMenu()

def saveFile():
	global editedImage
	global editedImageIndex
	if(editedImageIndex):
		cv2.imwrite(pydialog.saveFile().name,editedImage[editedImageIndex])

def openFile():
	global originalImage
	global editedImage
	global height,width,maxKernel

	global editedImageIndex,imgOpened
	if(imgOpened):
		editedImage.clear()
		editedImageIndex=0
		
	originalImage=cv2.imread(pydialog.getFile())
	editedImage+=(originalImage,)
	height=originalImage.shape[0]
	width=originalImage.shape[1]
	maxKernel=int(max(height,width)/2)
	imgOpened=True
	updateTrackbarPos()
	showEditedImage()
	setCurrentMenu(1)

def exitProgram():
	raise SystemExit

def passer():
	print('hi')
	
#################################################################################################	
#################################################################################################
#################################################################################################

#particular tiles:
def implementDenoise():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Denoise(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()


def implementNegative():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	r,g,b=cv2.split(img)
	r=255-r
	g=255-g
	b=255-b
	img=cv2.merge(([r,g,b]))
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementSpeckle():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Speckle(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()
	
def implementPoisson():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Poisson(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementSaltnPepper():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.SaltnPepper(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementGauss():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Gauss(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()
	
def implementCommon():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Common(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementLaplacian():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Laplacian(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementGray():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementHistEqualize():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.HistogramEqualization(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementSaturation(value):
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Saturation(img,value)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementHue(value):
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Hue(img,value)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementBrightness(value):
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Brightness(img,value)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementBilateral(ksize):
	ksize=(2*ksize)+1
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Bilateral(img,ksize)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementMedian(ksize):
	ksize=(2*ksize)+1
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Median(img,ksize)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementGaussian(ksize):
	ksize=(2*ksize)+1
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Gaussian(img,ksize)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementAveraging(ksize):
	ksize=(2*ksize)+1
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Averaging(img,ksize)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementRobert():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Robert(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementPrewitt():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Prewitt(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def implementSobel():
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Sobel(img)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

def CannyTh1(value):
	global Cannyth1
	global Cannyth2
	Cannyth1=value
	implementCanny(Cannyth1,Cannyth2)

def CannyTh2(value):
	global Cannyth1
	global Cannyth2
	Cannyth2=value
	implementCanny(Cannyth1,Cannyth2)

def implementCanny(th1,th2):
	global editedImage,editedImageIndex,flag
	img=editedImage[editedImageIndex-flag]
	img=imf.Canny(img,th1,th2)
	trueFlag(img) if flag else falseFlag(img) 
	showEditedImage()

#################################################################################################	
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

#Edition Window functions
# def startEdit():
# 	global editedImageIndex
# 	cv2.namedWindow('edition',cv2.WINDOW_NORMAL)
# 	cv2.resizeWindow('edition',sysWidth//2,sysHeight)
# 	cv2.createTrackbar('version','edition',0,editedImageIndex,startEdit)

def updateTrackbarPos():
# 	global editedImageIndex
# 	cv2.destroyWindow('edition')
# 	cv2.namedWindow('edition',cv2.WINDOW_NORMAL)
# 	cv2.resizeWindow('edition',sysWidth//2,sysHeight)
	# cv2.createTrackbar('version','edition',0,len(editedImage)-1,updateEdit)
	# cv2.setTrackbarPos('version','edition',editedImageIndex)
	showEditedImage()

def redo():
	global editedImageIndex
	global editedImage
	if(editedImageIndex<len(editedImage)-1):
		editedImageIndex+=1
		showEditedImage()

def undo():
	global editedImageIndex
	if(editedImageIndex>0):
		editedImageIndex-=1
		showEditedImage()

def updateEdit(value):
	global editedImageIndex
	editedImageIndex=value
	showEditedImage()

def showEditedImage():
	global editedImageIndex
	global editedImage
	global delIndex
	screenEdition()
#################################################################################################	
#################################################################################################
#################################################################################################
#################################################################################################	
#################################################################################################
#################################################################################################

#dialog window functions
def denoiseDialog():
	functionDialog()
	implementDenoise()

def negativeDialog():
	functionDialog()
	implementNegative()

def speckleDialog():
	functionDialog()
	implementSpeckle()

def poissonDialog():
	functionDialog()
	implementPoisson()

def saltnpepperDialog():
	functionDialog()
	implementSaltnPepper()

def gaussDialog():
	functionDialog()
	implementGauss()

def commonDialog():
	functionDialog()
	implementCommon()

def laplacianDialog():
	functionDialog()
	implementLaplacian()

def grayDialog():
	functionDialog()
	implementGray()

def histEqualizeDialog():
	functionDialog()
	implementHistEqualize()

def brightnessDialog():
	functionDialog()
	cv2.createTrackbar('dec','Dialog',256,511,implementBrightness)

def hueDialog():
	functionDialog()
	cv2.createTrackbar('value','Dialog',0,179,implementHue)

def saturationDialog():
	functionDialog()
	cv2.createTrackbar('dec','Dialog',256,511,implementSaturation)


def bilateralDialog():
	global maxKernel
	functionDialog()
	cv2.createTrackbar('ksize','Dialog',0,int(18),implementBilateral)

def medianDialog():
	global maxKernel
	functionDialog()
	cv2.createTrackbar('ksize','Dialog',0,int(40),implementMedian)

def gaussianDialog():
	global maxKernel
	functionDialog()
	cv2.createTrackbar('ksize','Dialog',0,int(0.1*maxKernel),implementGaussian)

def averagingDialog():
	global maxKernel
	functionDialog()
	cv2.createTrackbar('ksize','Dialog',0,int(0.5*maxKernel),implementAveraging)


def robertDialog():
	functionDialog()
	implementRobert()

def prewittDialog():
	functionDialog()
	implementPrewitt()

def sobelDialog():
	functionDialog()
	implementSobel()


def cannyDialog():
	functionDialog()
	global CannyTh1
	global CannyTh2
	cv2.createTrackbar('th1','Dialog',0,255,CannyTh1)
	cv2.createTrackbar('th2','Dialog',0,255,CannyTh2)

def closeDialog(x):
	global editionMade
	global flag
	global editedImageIndex
	global editedImage
	global delIndex

	if(x==1 and editionMade):
		del editedImage[len(editedImage)-1]
		editedImageIndex-=1
		if(delIndex>0):
			editedImageIndex=delIndex
		updateTrackbarPos()
	elif(editionMade and delIndex>0):
		print(1)
		img=editedImage[len(editedImage)-1]
		del editedImage[delIndex+2:len(editedImage)]
		editedImage[delIndex+1]=img
		editedImageIndex=delIndex+1
		delIndex=-1

	editionMade=0
	flag=0	
	cv2.destroyWindow('Dialog')

def functionDialog():
	global currentDialog
	cv2.namedWindow('Dialog',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Dialog',sysWidth//4,sysHeight//8)
	cv2.imshow('Dialog',menuTrackbar4[0])
	currentDialog=4
	cv2.setMouseCallback('Dialog',functionerDialog)
#################################################################################################	
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

def functionCaller(menuNo,fNo):
	if(menuNo==0):
		switcher={
		0:exceptionMenu,
		1:passer,
		2:passer,
		3:passer,
		4:passer,
		5:passer,
		6:passer,
		7:passer}
		return switcher.get(fNo)()
	if(menuNo==1):
		switcher={
		0:setCurrentMenu,
		1:setCurrentMenu,
		2:setCurrentMenu,
		3:setCurrentMenu,
		4:setCurrentMenu,
		5:goBack,
		20:2,
		21:3,
		22:5,
		23:6,
		24:7,
		6:passer}
		if fNo < 5:
			return switcher.get(fNo)(switcher.get(20+fNo))
		else:
			return switcher.get(fNo)()
	if(menuNo==2):
		switcher={
		0:cannyDialog,
		1:sobelDialog,
		2:prewittDialog,
		3:robertDialog,
		4:goBack,
		5:passer,
		6:passer,
		7:passer}
		return switcher.get(fNo)()
	if(menuNo==3):
		switcher={
		0:averagingDialog,
		1:gaussianDialog,
		2:medianDialog,
		3:bilateralDialog,
		4:goBack,
		5:passer,
		6:passer,
		7:passer}
		return switcher.get(fNo)()
	if(menuNo==4):
		switcher={
		0:closeDialog,
		1:closeDialog}
		return switcher.get(fNo)(fNo)
	if(menuNo==5):
		switcher={
		0:brightnessDialog,
		1:hueDialog,
		2:saturationDialog,
		3:histEqualizeDialog,
		4:negativeDialog,
		5:grayDialog,
		6:goBack,
		7:passer}
		return switcher.get(fNo)()
	if(menuNo==6):
		switcher={
		0:laplacianDialog,
		1:commonDialog,
		2:goBack,
		3:passer,
		4:passer,
		5:passer,
		6:passer,
		7:passer}
		return switcher.get(fNo)()
	if(menuNo==7):
		switcher={
		0:gaussDialog,
		1:saltnpepperDialog,
		2:speckleDialog,
		3:denoiseDialog,
		4:goBack,
		5:passer,
		6:passer,
		7:passer}
		return switcher.get(fNo)()
	if(menuNo==8):
		switcher={
		0:openFile,
		1:saveFile,
		2:undo,
		3:redo,
		4:toggleCompare,
		5:passer,
		6:exitProgram,
		}
		return switcher.get(fNo)()

def functionerMenu(event,x,y,flags,params):
	if(event==cv2.EVENT_LBUTTONDOWN):
		global xPad,yPad,xxPad,yyPad
		global currentMenu
		global change
		if(x<xPad):
			cM=8
			xP=xxPad
			yP=yyPad
		else:
			cM=currentMenu
			xP=xPad
			yP=yPad
		nF=len(menu[cM][1])
		for i in range(nF):
			if(change):
				i=1
				change=0
			a1=menu[cM][1][i][0][0]+xP
			a2=menu[cM][1][i][0][1]+yP
			b1=menu[cM][1][i][1][0]+xP
			b2=menu[cM][1][i][1][1]+yP
			if (a1<= x <=b1) and (a2<= y <=b2):
				functionCaller(cM,i)

def functionerDialog(event,x,y,flags,params):
	if(event==cv2.EVENT_LBUTTONDOWN):
		global currentDialog
		currentMenu=currentDialog
		nF=len(menu[currentMenu][1])
		for i in range(nF):
			a1=menu[currentMenu][1][i][0][0]
			a2=menu[currentMenu][1][i][0][1]
			b1=menu[currentMenu][1][i][1][0]
			b2=menu[currentMenu][1][i][1][1]
			if (a1<= x <=b1) and (a2<= y <=b2):
				functionCaller(currentMenu,i)
#################################################################################################	
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################		

#menu function calls		
# cv2.namedWindow('Menu',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Menu',sysWidth//2,sysHeight)
#cv2.setMouseCallback('Menu',functionerMenu)

cB=(255,0,0);cG=(0,255,0);cR=(0,0,255);cY=(0,255,255)
cO=(0,127,255);cP=(255,0,128);cLB=(255,255,0);cU=(128,20,230);cDG=colordarkgray

menuTrackbar4=mc.menuCreator(2,('okay','cancel'),(cG,cR),100,600)
menu0=mc.menuCreator(8,('Menu','','','','','','',''),(cB,cDG,cDG,cDG,cDG,cDG,cDG,cDG))
menu1=mc.menuCreator(8,('Edge Detection','Smoothing','Spatial','Sharpening','Noise','Back','',''),(cB,cP,cO,cG,cR,cY,cDG,cDG))
menu2=mc.menuCreator(8,('Canny','Sobel','Prewitt','Robert','Back','','','',''),(cG,cB,cP,cO,cLB,cDG,cDG,cDG))
menu3=mc.menuCreator(8,('Averaging','Gaussian','Median','Bilateral','Back','','',''),(cB,cP,cO,cU,cG,cDG,cDG,cDG))
menu5=mc.menuCreator(8,('Brightness','Hue','Saturation','Hist Equalize','Negative','Gray','Back',''),(cG,cP,cR,cU,cLB,cU,cY,cDG))
menu6=mc.menuCreator(8,('Laplacian','Common','Back','','','','',''),(cR,cP,cLB,cDG,cDG,cDG,cDG,cDG))
menu7=mc.menuCreator(8,('Gauss','SaltnPepper','Speckle','Denoise','Back','','',''),(cG,cY,cB,cO,cU,cDG,cDG,cDG))
menu8=mc.menuCreator(7,('Open','Save','Undo','Redo','Compare','','Exit'),(cG,cLB,cB,cY,cP,cDG,cR),50,screenImgWidth,False,True)
menu=(menu0,menu1,menu2,menu3,menuTrackbar4,menu5,menu6,menu7,menu8)

# cv2.imshow('Menu',menu[currentMenu][0])
screenMenu()
cv2.setMouseCallback('',functionerMenu)

#################################################################################################
#################################################################################################

#edition function calls

# startEdit()


#################################################################################################
#################################################################################################
cv2.waitKey(0)
cv2.destroyAllWindows()



