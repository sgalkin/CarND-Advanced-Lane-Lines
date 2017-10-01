import numpy as np
import cv2
import functools
import math

'''Combined threshold'''
def combined_threshold(img, hls):
    c_th = color_threshold(img, hls)
    s_th = sobel_threshold(img, hls)
    return bitwise_or(c_th, s_th)


'''Color threshold'''
def color_threshold(rgb, hls):
    HUE = (25, 35)
    SAT = (180, 200)

    h_th = threshold(hls[:,:,0], *HUE, mask=255)
    s_th = threshold(hls[:,:,2], *SAT, mask=255)
    return bitwise_or(h_th, s_th)


'''Sobel threshold'''
def sobel_threshold(rgb, hls):
    K = 23 # kernel
    
    SAT_XY = (20, 200)
    LUM_X = (20, 256)
    DIRECTION = tuple(math.pi*x/180. for x in (80, 100))

    lum_sx, lum_sy = sobel(hls[:,:,1], K)
    sat_sx, sat_sy = sobel(hls[:,:,2], K)

    mag_lum_x = normalize(magnitude(lum_sx))
    sth_lum_x = threshold(mag_lum_x, *LUM_X, mask=255)
    dth_lum_x = np.uint8(threshold(
        angle(np.absolute(lum_sy), np.absolute(lum_sx)), *DIRECTION, mask=255))

    mag_sat_xy = normalize(magnitude(sat_sx, sat_sy))
    sth_sat_xy = threshold(mag_sat_xy, *SAT_XY, mask=255)
    dth_sat_x = np.uint8(threshold(
        angle(np.absolute(sat_sy), np.absolute(sat_sx)), *DIRECTION, mask=255))

    return bitwise_or(bitwise_and(sth_lum_x, dth_lum_x),
                      bitwise_and(sth_sat_xy, dth_sat_x))


'''Simple threshold implementation'''
def threshold(img, thmin=0, thmax=256, mask=255):
    dimg = np.zeros_like(img)
    dimg[(img >= thmin) & (img < thmax)] = mask
    return dimg


def bitwise_or(*args):
    return functools.reduce(np.bitwise_or, args, 0)

def bitwise_and(*args):
    # TODO: make it work not only for uint8 :)
    return functools.reduce(np.bitwise_and, args, 255)


'''Sobel threshold support functions'''
def sobel(img, ksize):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return sobel_x, sobel_y

def magnitude(dx, dy=None):
    return np.absolute(dx) if dy is None else np.sqrt(np.square(dx) + np.square(dy))

def angle(dx, dy):
    return np.arctan2(dy, dx)

'''Normalization routine'''
def normalize(m, rmin=None, rmax=None):
    rmin = np.min(m) if rmin is None else rmin
    rmax = np.max(m) if rmax is None else rmax
    return np.uint8(255.*(m - rmin)/(rmax - rmin))

