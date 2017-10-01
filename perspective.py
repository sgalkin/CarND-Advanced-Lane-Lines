import cv2
import numpy as np

from cv_utils import shape_to_size

def get_road_borders():
    TOP = 450
    BOTTOM = 685
    return TOP, BOTTOM

def get_road_perspective_transformation(shape):
    SCALE = 0.75
    
    TD = 37 # top delta
    TXR = 10 # top extra delta for the right lane
    BD = 379 # bottom delta
    BXR = 146 # bottom extra delta for the right lane
    
    half_x = shape[1]//2
    top, bottom = get_road_borders()

    src = np.array([
            [half_x - TD, top],
            [half_x + TD + TXR, top],
            [half_x + BD + BXR, shape[0]],
            [half_x - BD, shape[0]],
        ], dtype=np.float32)

    dst = np.array([
            [half_x - BD*SCALE, 0],
            [half_x + (BD + BXR)*SCALE, 0],
            [half_x + (BD + BXR)*SCALE, 2*shape[0]],
            [half_x - BD*SCALE, 2*shape[0]],
        ], dtype=np.float32)

    Mp = cv2.getPerspectiveTransform(src, dst)
    MpInv = cv2.getPerspectiveTransform(dst, src)

    def forward(image):
        osize = shape_to_size(shape)
        dsize = (osize[0], 2*osize[1])
        return cv2.warpPerspective(image, Mp, dsize=dsize, flags=cv2.INTER_LINEAR)
    
    def backward(image):
        dsize = shape_to_size(shape)
        return cv2.warpPerspective(image, MpInv, dsize=dsize, flags=cv2.INTER_LINEAR)
    
    return forward, backward

if __name__ == '__main__':
    import cv2
    import camera
    M, d = camera.load('camera.p')
    calibration = cv2.imread('test_images/straight01.jpg')
    calibration = cv2.undistort(calibration, M , d)
    fw, _ = get_road_perspective_transformation(calibration.shape)
    cv2.imwrite('calibration.jpg', fw(calibration))
