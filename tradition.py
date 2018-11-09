import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('./datasets/synthetic_shapes_v6/draw_checkerboard/images/training/138.png',0)

    # Initiate feature detectors
    orb = cv2.ORB_create()
    brisk = cv2.BRISK_create()
    kaze = cv2.KAZE_create()

    # find and keypoints
    kp_orb = orb.detect(img, None)
    kp_brisk = brisk.detect(img, None)
    kp_kaze = kaze.detect(img, None)
    
    # draw the keypoints
    img_orb = img.copy()
    img_orb = cv2.drawKeypoints(img, kp_orb, img_orb, color=(255,0,0))
    img_brisk = img.copy()
    img_brisk = cv2.drawKeypoints(img, kp_brisk, img_brisk, color=(255,0,0))
    img_kaze = img.copy()
    img_kaze = cv2.drawKeypoints(img, kp_kaze, img_kaze, color=(255,0,0))


    cv2.imshow('With BRISK',img_brisk)
    cv2.waitKey(0)
    cv2.imshow('With ORB',img_orb)
    cv2.waitKey(0)
    cv2.imshow('With KAZE',img_kaze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':    
    main()
