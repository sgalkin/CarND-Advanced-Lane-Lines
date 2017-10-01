import numpy as np
import functools
import cv2

MARGIN = 80 # The width of the windows +/- margin

'''Fits polynomial of given degree'''
def fit_poly(img, degree, detector, x_scale, y_scale):
    def active_points(img):
        # Identify the x and y positions of all nonzero pixels in the image
        return [np.array(x) for x in img.nonzero()][::-1]
    
    lx, ly, rx, ry, o = detector(img, *active_points(img))
    if len(lx) == 0 or len(rx) == 0:
        return None, None, o

    o[ly, lx] = [255, 0, 0]
    o[ry, rx] = [0, 0, 255]
    
    lp, rp = (np.polyfit(y*y_scale, x*x_scale, degree) for x, y in ((lx, ly), (rx, ry)))

    o = plot_poly(o, lp, rp, x_scale, y_scale)
    return lp, rp, o


'''Detector factories'''
def sliding_window_detector(left_max=None, right_max=None):
    def analyze_hist(img):
        hist = np.sum(img[img.shape[0]//2:,:], axis=0)
        mid = hist.shape[0]//2
        left_max = np.argmax(hist[:mid])
        right_max = np.argmax(hist[mid:]) + mid  
        return left_max, right_max

    def passthrough(_):
        return left_max, right_max

    max_points = analyze_hist if left_max is None or right_max is None else passthrough
    return lambda img, ax, ay: sliding_window(img, *max_points(img), ax, ay)

def polynomial_detector(left_poly, right_poly, x_scale, y_scale):
    return lambda img, ax, ay: poly_window(img, left_poly, right_poly, ax, ay, x_scale, y_scale)


'''Simple sliding window'''
def sliding_window(img, left_max, right_max, active_x, active_y):
    N_WINDOWS = 16 # Number of windows
    MIN_PIX = 100  # Minimum number of pixels found to recenter window

    def check_window(x, high_y, low_y, active_x, active_y):
        # Identify the nonzero pixels in x and y within the window
        return ((active_y >= low_y) & (active_y < high_y) &
                (active_x >= x - MARGIN) & (active_x < x + MARGIN)).nonzero()[0]

    assert(len(img.shape)==2)
    out = np.dstack(3*[img])
  
    lpos, rpos = np.int(left_max), np.int(right_max)
    lidx, ridx = [], []

    height = img.shape[0]//N_WINDOWS

    # Step through the windows one by one
    for w in range(N_WINDOWS):
        high_y = img.shape[0] - w*height
        low_y = high_y - height

        # Identify window boundaries in x and y (and right and left)
        l, r = (check_window(side, high_y, low_y, active_x, active_y)
                for side in (lpos, rpos))

        functools.reduce(
            lambda i, x: cv2.rectangle(i,
                                       (x - MARGIN, low_y),
                                       (x + MARGIN, high_y), 
                                       thickness=3, color=(0, 255, 0)), 
            (lpos, rpos), out)
       
        # Append these indices to the lists
        lidx.append(l)
        ridx.append(r)
    
        # If you found > minpix pixels, recenter next window on their mean position
        if len(l) > MIN_PIX:
            lpos = np.int(np.mean(active_x[l]))
            
        if len(r) > MIN_PIX:        
            rpos = np.int(np.mean(active_x[r]))
            
    # Concatenate the arrays of indices
    lidx, ridx = np.concatenate(lidx), np.concatenate(ridx)

    # Extract left and right line pixel positions
    lx, ly = active_x[lidx], active_y[lidx] 
    rx, ry = active_x[ridx], active_y[ridx]

    return lx, ly, rx, ry, out


'''Polynomial window. 
The idea is to grab pixel within margin from previous approximation'''
def poly_window(img, left_poly, right_poly, active_x, active_y, x_scale, y_scale):
    assert(len(img.shape)==2)
    out = np.dstack(3*[img])
    
    l, r = evaluate(active_y, left_poly, right_poly, x_scale, y_scale)
    lidx, ridx = ((active_x > (p - MARGIN)) & (active_x < (p + MARGIN)) for p in (l, r))

    lx, ly = active_x[lidx], active_y[lidx]
    rx, ry = active_x[ridx], active_y[ridx]

    cv2.fillPoly(out, make_polygon(active_y, l - MARGIN, l + MARGIN), color=(0, 255, 0))
    cv2.fillPoly(out, make_polygon(active_y, r - MARGIN, r + MARGIN), color=(0, 255, 0))

    return lx, ly, rx, ry, out


def evaluate(y, lp, rp, x_scale, y_scale):
    return (np.polyval(c, y*y_scale)/x_scale for c in (lp, rp))


'''Drawing helpers'''
def make_polylines(y, left, right):
    return [np.int32(np.transpose(np.vstack((side, y))))
            for side in (left, right)]

def make_polygon(y, left, right):
    l, r = make_polylines(y, left, right)
    return [np.vstack((l, np.flipud(r)))]

def plot_area(shape, left_poly, right_poly, x_scale, y_scale):
    y = np.linspace(0, shape[0]-1, shape[0])
    l, r = evaluate(y, left_poly, right_poly, x_scale, y_scale)
    return area(shape, l, r)

def plot_area(shape, left, right):
    y = np.linspace(0, shape[0]-1, shape[0])
    out = np.zeros(list(shape) + ([3] if len(shape) == 2 else []), dtype=np.uint8)
    cv2.fillPoly(out, make_polygon(y, left, right), color=(0, 255, 0))
    return out

def plot_poly(img, left_poly, right_poly, x_scale, y_scale):
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    l, r = evaluate(y, left_poly, right_poly, x_scale, y_scale)
    cv2.polylines(img, make_polylines(y, l, r),
                  isClosed=False, color=(255,255,255), thickness=8)
    return img

