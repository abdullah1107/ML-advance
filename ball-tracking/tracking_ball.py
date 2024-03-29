# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

#define the lower and upperboundary of tracking ball color
#ball in the HSV color space, then initialize the list of tracked points

green_lower=(29, 86, 6)
green_upper=(64, 255, 255)
pts = deque(maxlen=args["buffer"])

#if a video path not supplied, grab the reference to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

#if grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

#allow the camera or video file to warm up
time.sleep(2.0)

#keep looping
while True:
    #grab the current frame
    frame = vs.read()
    #handle the frame from VideoStream or VideoCapture
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    #resize the freame, blur it and convert it to the HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #construct a mask for the color "green" then perform
    #a series of dilations and erosions to remove any small
    #blobs left in the mask

    mask = cv2.inRange(hsv, green_lower,  green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #find contours in the mask and initialize the current
    #(x,y) center of the ball

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    #only proceed if at least one contour was found
    if len(cnts)>0:
        #find the largest contour in the mask, then use it to compute the minimum enclosing circle and cntroid
        c = max(cnts, key = cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:

            cv2.circle(frame, (int(x), int(y)), int(radius),
            (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    #update the points queue
    pts.appendleft(center)

    #loop over the set of tracked points
    for i in range(1, len(pts)):
        #if either of the tracked points are none, ignore
        if (pts[i-1] is None or pts[i] is None):
            continue
        #otherwise compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float (i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break


if not args.get("video", False):
    vs.stop()
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
