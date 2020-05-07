import cv2
import numpy as np 
colordarkgray=(80,65,69)

def menuCreator(tabs=2,text=(('abc'),('abc')),color=((255,255,255),(255,255,255)),menuHeight=635,menuWidth=410,flag=True,oneRow=False):

	pointArray=[]
	menu=np.zeros((menuHeight,menuWidth,3),np.uint8)
	if(flag):
		if(tabs%2==1):
			text=text+('',)
			color=color+(colordarkgray,)
		
		
		rows=(((tabs+1)//2))
		cols=2


		tabWidth=menuWidth/2
		tabHeight=menuHeight/rows
	else:
		if(oneRow):
			cols=tabs
			rows=1
		else:
			rows=tabs
			cols=1
		tabWidth=menuWidth/cols
		tabHeight=menuHeight/rows



	for i in range(rows):
		for j in range(cols):
			x=int(j*tabWidth)
			y=int(i*tabHeight)
			x1=int((j+1)*tabWidth)
			y1=int((i+1)*tabHeight)
			b,g,r=(color[(i*cols)+j])
			pointArray.append(((x,y),(x1,y1)))
			textsize = cv2.getTextSize(text[(i*2)+j], cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
			textX = (tabWidth - textsize[0]) // 2
			textY = (tabHeight + textsize[1]) // 2
		
			cv2.rectangle(menu,(x,y),(x1,y1),color[(i*2)+j],-1)
			cv2.rectangle(menu,((x),(y)),((x1),(y1)),colordarkgray,10)
			cv2.putText(menu,text[(i*2)+j],(int(x+textX),int(y+textY)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255-b,255-g,255-r),2)
	return menu,pointArray


	