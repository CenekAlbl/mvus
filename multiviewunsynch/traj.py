import cv2
import numpy as np
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", type=str, help="path to input video file")
# ap.add_argument("-o", "--output", type=str, help="output txt file name (.txt)")
# args = vars(ap.parse_args())

# inputVideoName = args["input"]
# outputFileName = args["output"]

inputVideoName = 'C:\\Users\\tong2\\MyStudy\\ETH\\2019FS\\Thesis\\data\\fixposition/c1_f1.MOV'
outputFileName = './output.txt'

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
#
# for i in range(bs_use_frames):
#     ret, frame = video.read()
#     if ret == False:
#         exit("not enough frames for background subtraction phase: %d\n" % bs_use_frames)
#     frame_read += 1
#
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

ret, frame = video.read()

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

    if ret == False:
        print("Video ending, press 'q' to close window.")
        while True:
            key = cv2.waitKey(60)
            if key == ord("q"):
                break
    frame_read += 1

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    key = cv2.waitKey(60)
    if key == ord("q"):
        break
    elif key == ord("s"):
        print("selecting target\n")
        initBB = cv2.selectROI("Frame", frame, fromCenter = False, showCrosshair = True)
        (x_pre, y_pre, width, height) = [int(v) for v in initBB]
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
            print("starting point selected: ", x_pre, y_pre)
            f.write("start with picked point: (%d, %d)\n" % (x_pre + width/2, y_pre + height/2))

        if secondBB is not None:
            _, track_window = cv2.meanShift(mask, (x_pre, y_pre, width, height), term_criteria)
            x_post, y_post, width, height = track_window
            cv2.rectangle(frame, (x_post, y_post), (x_post + width, y_post + height), (0, 255, 0), 2)
            traj = cv2.line(traj, (x_pre, y_pre), (x_post, y_post), color, 2)
            frame_draw += 1
            f.write("%d %d\n" % (x_post + width/2, y_post + width/2))
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
