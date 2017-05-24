from collections import deque
import numpy as np
from pyimagesearch import imutils
import cv2


Lower=(125,40,40)
Upper=(155,255,255)
pts=deque(maxlen=25)

camera=cv2.VideoCapture(0)

while True:
	(grabbed,frame)=camera.read()
	frame=imutils.resize(frame,width=600)
	#frame=cv2.GaussianBlur(frame,(11,11),0)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	
	mask=cv2.inRange(hsv,Lower,Upper)
	
	mask=cv2.erode(mask,None,iterations=3)
	mask=cv2.dilate(mask,None,iterations=3)
	

	cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center=None
	if len(cnts)>0:
		c=max(cnts,key=cv2.contourArea)
		((x,y),radius)=cv2.minEnclosingCircle(c)
		M=cv2.moments(c)
		center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
		if radius>10:
			cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
			cv2.circle(frame,center,5,(0,0,255),-1)
	pts.appendleft(center)
	for i in range(1,len(pts)):
		if pts[i-1] is None or pts[i] is None:
			continue
		cv2.line(frame,pts[i-1],pts[i],(0,0,255),2)
	cv2.imshow("Frame",frame)
			
	key=cv2.waitKey(1) & 0xFF
	if(key==ord('q')):
		break
camera.release()
cv2.destroyAllWindows
