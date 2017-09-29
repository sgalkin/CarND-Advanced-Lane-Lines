#!/usr/bin/env python

import camera

import numpy as np
import cv2

import argparse
import os
import functools
import itertools
from glob import glob

PATTERN_SIZE = (9, 6)
PATTERN_POINTS = np.float32(
    np.mgrid[:PATTERN_SIZE[0], :PATTERN_SIZE[1], :1].T.reshape(-1, 3))

def split_images(images):
    images = glob(os.path.join(images, '*.jpg'))
    return images[1:], [images[0]]

def get_image_points(names):
    failed = []
    img_objs = []

    for name, image in ((n, cv2.imread(n)) for n in names):
        r, corners = cv2.findChessboardCorners(image, PATTERN_SIZE,
                                               flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if r:
            img_objs.append(corners)
        else:
            failed.append(name)
        
    return np.vstack(np.expand_dims(img_objs, 0)), failed

def get_object_points(image_points):
    return np.vstack(
        itertools.repeat(np.expand_dims(PATTERN_POINTS, axis=0), len(image_points)))

def validate(Mc, dist, names, destination):
    for n in names:
        cv2.imwrite(
            os.path.join(destination, os.path.basename(n)),
            cv2.undistort(cv2.imread(n), Mc, dist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration utility')
    parser.add_argument('images', type=str,
                        help='Path to directory with calibration images')
    parser.add_argument('-o', dest='filename', type=str, required=True,
                        help='filename for calibration data')
    parser.add_argument('-g', dest='validate', type=str,
                        help='directory for generated validation images')

    args = parser.parse_args()
    c_set, v_set = split_images(args.images)
    print('Validation image(s):', ', '.join(v_set))
    image_points, failed = get_image_points(c_set)
    if len(failed) != 0:
        print('Failed to find image points:', ', '.join(failed))
    v_set += failed
    object_points = get_object_points(image_points)
    assert(len(image_points) == len(object_points))

    shape = cv2.imread(v_set[0]).shape[-2::-1]
    _, M, d, _, _ = cv2.calibrateCamera(object_points, image_points, shape, None, None)
    camera.store(args.filename, M, d)
    
    if args.validate:
        os.makedirs(args.validate, exist_ok=True)
        validate(M, d, v_set, args.validate)
