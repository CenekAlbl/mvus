import numpy as np
import cv2

def getFrame(video,numFrame):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES,numFrame)
    img = cap.read()[1]
    
    cap.release()
    return img

if __name__ == "__main__":
    video = 'C:\\Users\\tong2\\MyStudy\\ETH\\2019FS\\Thesis\\data\\C0028.MP4'
    img = getFrame(video,50)
    cv2.imshow('Frame',img)
    cv2.waitKey()