from collections import deque
import numpy as np
import cv2
from pyimagesearch import imutils

lower=(105,50,50)
upper=(125,243,216)
pts=deque(maxlen=60)

camera=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')#output the file somewhere fourcc is the codec
out=cv2.VideoWriter('output3.avi',fourcc,20.0,(640,480))
while True:
	(grabbed,frame)=camera.read()
	frame=imutils.resize(frame,width=600)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask=cv2.inRange(hsv,lower,upper)
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
			cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,0),2)
			cv2.circle(frame,center,5,(0,255,0),-1)
	pts.appendleft(center)
	for i in range(1,len(pts)):
		if pts[i-1] is None or pts[i] is None:
			continue
		cv2.line(frame,pts[i-1],pts[i],(0,255,0),2)
	out.write(frame)		
	cv2.imshow("Frame",frame)	
	k=cv2.waitKey(1) & 0xFF
	if k==ord('q'):
		break
camera.release()
out.release()
cv2.destroyAllWindows()

