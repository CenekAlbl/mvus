import cv2
import numpy as np

cv2.namedWindow("mask",0);
cv2.resizeWindow("mask", 480, 320);
cv2.namedWindow("template_matching",0);
cv2.resizeWindow("template_matching", 480, 320);
cv2.namedWindow("trajectory", 0)
cv2.resizeWindow("trajectory", 720, 540)

f = open("video_2_output.txt", "w+")

video = cv2.VideoCapture("video_2.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

frame_read = 0
frame_draw = 0
bs_use_frames = 30

for i in range(bs_use_frames):
    ret, frame = video.read()
    if ret == False:
        exit("not enough frames for background subtraction phase: %d\n" % bs_use_frames)
    frame_read += 1

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

x_pre = 1060
y_pre = 650
f.write("start with initial picked point: (%d, %d)\n" % (x_pre, y_pre))

width = 30
height = 30
roi = fgmask[y_pre: y_pre + height, x_pre: x_pre + width]
cv2.imshow("first_template", roi)
roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

traj = np.zeros_like(frame)
color = (0, 0, 255)

while True:
    ret, frame = video.read()

    if ret == False:
        print("video ending")
        break
    frame_read += 1

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    mask = cv2.calcBackProject([fgmask], [0], roi_hist, [0, 256], 1)

    mask = cv2.subtract(255, mask)

    _, track_window = cv2.meanShift(mask, (x_pre, y_pre, width, height), term_criteria)
    x_post, y_post, w, h = track_window
    cv2.rectangle(frame, (x_post, y_post), (x_post + w, y_post + h), (0, 255, 0), 2)

    traj = cv2.line(traj, (x_pre, y_pre), (x_post, y_post), color, 2)

    cv2.imshow("mask", mask)
    cv2.imshow("template_matching", frame)
    cv2.imshow("trajectory", traj)
    frame_draw += 1
    f.write("%d %d\n" % (x_post, y_post))

    x_pre = x_post
    y_pre = y_post

    key = cv2.waitKey(60)
    if key == 27:
        break

f.write("frames used for background subtraction (only in the beginning): %d\n" % bs_use_frames)
f.write("read in frames: %d\n" % frame_read)
f.write("draw out frames: %d\n" % frame_draw)
f.close()
video.release()
cv2.destroyAllWindows()
