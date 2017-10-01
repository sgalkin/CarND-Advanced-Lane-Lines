import numpy as np

class Averager:
    def __init__(self, length, height):
        self._avg = np.empty(shape=(height, length))
        self._counter = 0

    def push(self, vec):
        self._avg[:,self._index(self._counter)] = vec
        self._counter += 1

    def latest(self):
        return self._avg[:,self._index(self._counter - 1)]

    def mean(self):
        idx = min(self._counter, self._avg.shape[-1])
        return np.mean(self._avg[:,:idx], axis=1)

    def _index(self, n):
        return n % self._avg.shape[-1]


if __name__ == '__main__':
    a = Averager(4, 2)
    a.push([1, 3])
    assert((a.latest() == a.mean()).all())
    assert((a.latest() == [1, 3]).all())

    a.push([3, 1])
    assert((a.latest() == [3, 1]).all())
    assert((a.mean() == [2, 2]).all())

    for i in range(6):
        a.push([1, 4])
    assert((a.latest() == [1, 4]).all())
    assert((a.mean() == [1, 4]).all())

