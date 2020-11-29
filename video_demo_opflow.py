import cv2
import imutils
import argparse
import numpy as np
from imutils.video import FPS
from matplotlib import pyplot as plt
from visualize_cv2 import model, display_instances, class_names
import sys
import dlib
import os

args = sys.argv
if(len(args) < 2):
	print("run command: python video_demo.py 0 or video file name")
	sys.exit(0)
name = args[1]
if(len(args[1]) == 1):
	name = int(args[1])	
stream = cv2.VideoCapture(name)
fps = FPS().start()
frame_width = int(stream.get(3))
frame_height = int(stream.get(4))
list_all=[]
list0 = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while True:
	ret , frame = stream.read()
	if not ret:
		print("unable to fetch frame")
		break
	results = model.detect([frame], verbose=1)
	# Visualize results
	r = results[0]
	#print(r)
	masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'], list_all, list1, list2, list3, list4, list5, list6)
	list_all=masked_image[0]
	list1=masked_image[0]
	list2=masked_image[0]
	list3=masked_image[0]
	list4=masked_image[0]
	list5=masked_image[0]
	list6=masked_image[0]
	#print("coba {}".format(list_all))
	out.write(masked_image[7])	
	#cv2.imshow("masked_image",masked_image)
	#cv2.show()
	fps.update()
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Rata rata FPS: {:.2f}".format(fps.fps()))
stream.release()
out.release()
#cv2.destroyWindow("masked_image")
