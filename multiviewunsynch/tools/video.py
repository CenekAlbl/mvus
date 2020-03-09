# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import cv2


def getFrame(video,numFrame):
    '''
    This function reads a video and a list of frame numbers,
    returns then the frames as a list of images
    '''

    imgs = []
    cap = cv2.VideoCapture(video)
    for i in numFrame:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        img = cap.read()[1]
        imgs.append(img)
    
    cap.release()
    return imgs


def play_two_videos(v1,v2,f1,f2,t):
    '''
    Currently not working !!
    '''

    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)

    for i in range(t):
        # cap1.set(cv2.CAP_PROP_POS_FRAMES,f1+t)
        # cap2.set(cv2.CAP_PROP_POS_FRAMES,f2+t)

        read1,img1 = cap1.read()
        read2,img2 = cap2.read()

        if not read1 or not read2:
            print("Can't receive frame. Exiting...")
            break
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow('video1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('video2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('video1',500,300)
        cv2.resizeWindow('video2',500,300)
        cv2.imshow('video1', gray1)
        cv2.imshow('video2', gray2)
        if cv2.waitKey(1000 // 12) & 0xFF == 27:
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = 'C:\\Users\\tong2\\MyStudy\\ETH\\2019FS\\Thesis\\data\\first_flight/cam1_flight_1.MP4'
    img = getFrame(video,[60*248])
    cv2.imshow('Frame',img[0])
    cv2.waitKey()