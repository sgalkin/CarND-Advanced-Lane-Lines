import cv2
import numpy as np

def show(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

# position of right top angle (sorry)
def image_in_image(host, img, scale=4, ypos=20, xpos=20):
    rimg = cv2.resize(img,
                      dsize=tuple(x//scale for x in shape_to_size(img.shape)))
    host[ypos:ypos+rimg.shape[0],
         host.shape[1] - xpos - rimg.shape[1]:host.shape[1] - xpos
    ] = rimg
    return host

def shape_to_size(shape):
    return shape[-2::-1] if len(shape) == 3 else shape[::-1]


def adjust_image(rgb, A, B):
    f = np.float32(rgb)
    m = np.mean(np.mean(f, axis=0), axis=0)
    return np.uint8(np.maximum(0, np.minimum(255, A*(f - m) + m + B)))


def rgb2hls(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)

def rgb2hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

def rgb2gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


if __name__ == '__main__':
    assert(shape_to_size((10, 15, 3)) == (15, 10))
    assert(shape_to_size((10, 15)) == (15, 10))
