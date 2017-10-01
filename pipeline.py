from perspective import get_road_perspective_transformation
from threshold import combined_threshold
from fit_policy import FitPolicy
from averager import Averager

import camera
import cv_utils
import fit
import mpp

import cv2
import numpy as np
import functools
import math

ADJUST_CONTRAST_FACTOR = 1.5
ADJUST_BRIGHTNESS = -5

POLY_DEGREE = 2 # polynomial degree
AVERAGER_DEPTH = 4 # number of samples in averager
TOLERANCE = 1 # number of poor fits before fallback to sliding window

class Pipeline:
    def __init__(self, filename, shape):
        self._Mc, self._dist = camera.load(filename)
        self._forward, self._backward = get_road_perspective_transformation(shape)
        self._original_shape = shape
        self._warped_shape = self._forward(np.zeros(shape)).shape

        self._left_averager = Averager(AVERAGER_DEPTH, self._warped_shape[0])
        self._right_averager = Averager(AVERAGER_DEPTH, self._warped_shape[0])
        self._policy = FitPolicy(TOLERANCE, mpp.SCALE)

    def apply(self, rgb):
        undist = cv2.undistort(rgb, self._Mc, self._dist)
        warped = self._forward(undist)
        adjusted = cv_utils.adjust_image(warped, ADJUST_CONTRAST_FACTOR, ADJUST_BRIGHTNESS)
        threshold = combined_threshold(adjusted, cv_utils.rgb2hls(adjusted))
        radius, position, fitted, area = self._fit(threshold)
        unwarped = self._backward(area)
        combined = cv2.addWeighted(undist, 1.0, unwarped, 0.2, 0)
        annotated = self._annotate(combined, radius, position)
        return cv_utils.image_in_image(annotated, fitted, xpos=20, ypos=20)

    def _annotate(self, img, radius, position):
        cv2.putText(img, 'Curvature radius: {:=08.1f}m'.format(radius),
                    (20, 60), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color=(255, 255, 0))

        cv2.putText(img, 'Vehicle position: {} {:=03.0f}cm'.format(
                    '<=' if position < 0 else '=>', math.fabs(position)*100),
                    (20, 90), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color=(255, 255, 0))
        return img
    
    def _fit(self, wp):
        detector, acceptor = next(self._policy)
        lp, rp, o = fit.fit_poly(wp, POLY_DEGREE, detector, *mpp.SCALE)
        if lp is not None and rp is not None:
            y = np.linspace(0, wp.shape[0]-1, wp.shape[0])
            l, r = fit.evaluate(y, lp, rp, *mpp.SCALE)
            la, ra = acceptor(l, r)
        else:
            la, ra = False, False
            lp, rp = 2*[(POLY_DEGREE + 1)*[0]]

        if la: self._left_averager.push(l)
        if ra: self._right_averager.push(r)

        l_base = self._left_averager.latest() if la else self._left_averager.mean()
        r_base = self._right_averager.latest() if ra else self._right_averager.mean()

        if all((la, ra)):
            self._policy.accepted(l_base, r_base, lp, rp)
        else:
            self._policy.rejected(l_base, r_base)

        return (
            curvature(wp.shape, lp, rp, *mpp.SCALE),
            position(wp.shape, lp, rp, *mpp.SCALE),
            o,
            fit.plot_area(wp.shape, l_base, r_base))

    
def curvature(shape, l_poly, r_poly, x_scale, y_scale):
    assert(POLY_DEGREE == 2) # TODO make it generic
    return np.mean([
             np.mean([
                ((1+(2*poly[0]*(shape[0]-d)*y_scale + poly[1])**2)**1.5)/np.absolute(2*poly[0])
                for d in range(10)])
             for poly in (l_poly, r_poly)])

def position(shape, l_poly, r_poly, x_scale, y_scale):
    lane_center = np.mean([
                   np.mean([
                       np.polyval(p, [(shape[0]-d)*y_scale])
                       for d in range(10)])
                    for p in (l_poly, r_poly)])
    
    frame_center = (shape[1]//2)*x_scale
    return frame_center - lane_center

if __name__ == '__main__':
    img = cv2.imread('test_images/straight01.jpg')
    p = Pipeline('camera.p', img.shape)
    o = p.apply(img)
    cv_utils.show(o)
