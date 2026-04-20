import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import cv2

from test_realsense import RealsenseAPI

def read_referrence():
    # read from the hdf5 file and get one image for refference
    f = h5py.File('data/leap_clean.hdf5', 'r')

    img = f['data/demo_1/obs/agentview_image'][0]

    
    return img


def compare():
    ref = read_referrence()
    window_size = 1280
    ref = cv2.resize(ref, (window_size, window_size)).astype(np.uint8)

    cam = RealsenseAPI(device_id='123622270882')
    cam.connect()
    while True:
        cur = cam.get_rgbd()[:, :, :3]
        cur = cv2.resize(cur[:, 80:-80], (window_size, window_size)).astype(np.uint8)

        
        cv2.imshow('ref', ref)
        cv2.imshow('cur', cur)

        # overlay the images
        overlay = cv2.addWeighted(ref, 0.5, cur, 0.5, 0)

        cv2.imshow('overlay', overlay.astype(np.uint8))

        cv2.waitKey(1)

if __name__ == '__main__':
    compare()