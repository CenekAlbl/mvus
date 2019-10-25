import cv2
import numpy as np
import argparse

inputVideoName = 'C:/Users/tong2/Desktop/dronetracking/06_09_2019_6cams/videos/gopro3.MP4'
outputFileName = './data/icra/detection_gopro.txt'

cv2.namedWindow("Mask",0)
cv2.resizeWindow("Mask", 480, 320)
cv2.namedWindow("Frame",0)
cv2.resizeWindow("Frame", 480, 320)
cv2.namedWindow("Trajectory", 0)
cv2.resizeWindow("Trajectory", 720, 540)

f = open(outputFileName, "w+")

video = cv2.VideoCapture(inputVideoName)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

frame_read = 0
frame_draw = 0


term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

ret, frame = video.read()
# frame = cv2.flip(frame, -1)
frame_read += 1
traj = np.zeros_like(frame)
color = (0, 0, 255)

initBB = None
secondBB = None

x_pre = 0
y_pre = 0
width = 0
height = 0

startTracking = False
while True:
    ret, frame = video.read()
    # frame = cv2.flip(frame, -1)
    frame_read += 1

    if ret == False:
        print("Video ending, press 'q' to close window.")
        while True:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("background", fgmask)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        print("selecting target\n")
        initBB = cv2.selectROI("Frame", frame, fromCenter = False, showCrosshair = True)
        (x_pre, y_pre, width, height) = [int(v) for v in initBB]
        if width == 0 and height == 0:
            initBB = None
            print("cancel selection")
        if initBB is not None:
            print("target selected: ", x_pre, y_pre)
            roi = fgmask[y_pre: y_pre + height, x_pre: x_pre + width]
            cv2.imshow("first_template", roi)
            roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
            roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    if initBB is not None:
        mask = cv2.calcBackProject([fgmask], [0], roi_hist, [0, 256], 1)
        mask = cv2.subtract(255, mask)
        if key == ord("k"):
            print("selecting track starting point\n")
            secondBB = cv2.selectROI("Mask", mask, fromCenter = False, showCrosshair = True)
            (x_pre, y_pre, width, height) = [int(v) for v in secondBB]
            if width == 0 and height == 0:
                secondBB = None
                print("cancel selection")
            if secondBB is not None:
                print("new picked point: (%d, %d) at frame: %d\n" % (x_pre + width/2, y_pre + height/2, frame_read))
        if key == ord("t"):
            secondBB = None
            print("stop tracking at frame: %d" % (frame_read))


        if secondBB is not None:
            _, track_window = cv2.meanShift(mask, (x_pre, y_pre, width, height), term_criteria)
            x_post, y_post, width, height = track_window
            cv2.rectangle(frame, (x_post, y_post), (x_post + width, y_post + height), (0, 255, 0), 2)
            traj = cv2.line(traj, (x_pre, y_pre), (x_post, y_post), color, 2)
            frame_draw += 1
            f.write("%d %d %d\n" % (x_post + width/2, y_post + width/2, frame_read))
            x_pre = x_post
            y_pre = y_post

        cv2.imshow("Mask", mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Trajectory", traj)




f.write("read in frames: %d\n" % frame_read)
f.write("draw out frames: %d\n" % frame_draw)
f.close()
video.release()
cv2.destroyAllWindows()
