from fit import sliding_window_detector, polynomial_detector
from sklearn.metrics import mean_squared_error

MSE_THRESHOLD = 1000

class FitPolicy:
    def __init__(self, tolerance, scale):
        def mse_threshold(l, r, base_l, base_r, th):
            return (mean_squared_error(*x) < th for x in ((l, base_l), (r, base_r)))
        
        self._max_tolerance = tolerance
        self._tolerance = 0
        self._l_base = [None]
        self._r_base = [None]
        self._l_poly = None
        self._r_poly = None

        self._scaled_polynomial_detector = lambda lp, rp: polynomial_detector(lp, rp, *scale)
        self._mse_threshold_acceptor = lambda lb, rb: lambda l, r:  mse_threshold(l, r, lb, rb, MSE_THRESHOLD)
        self._force_acceptor = lambda l, r: (True, True)

    def __next__(self):
        if self._tolerance <= 0:
            self._tolerance = self._max_tolerance
            return (
                sliding_window_detector(self._l_base[-1], self._r_base[-1]),
                self._force_acceptor
            )
        else:
            return (
                self._scaled_polynomial_detector(self._l_poly, self._r_poly),
                self._mse_threshold_acceptor(self._l_base, self._r_base)
            )

    def accepted(self, l_base, r_base, l_poly, r_poly):
        self._tolerance = min(self._max_tolerance, self._tolerance + 1)
        self._l_base, self._r_base = l_base, r_base
        self._l_poly, self._r_poly = l_poly, r_poly

    def rejected(self, l_base, r_base):
        self._tolerance -= 1
        self._l_base, self._r_base = l_base, r_base
            
        
if __name__ == '__main__':
    fp = FitPolicy(1, (1, 1))
    d, a = next(fp)
    fp.rejected([1], [2])
    next(fp)
    fp.accepted([1], [2], [3], [4])
    next(fp)
    
