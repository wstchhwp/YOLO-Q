from collections import deque

class CouterMeter:
    def __init__(self, window_size=50):
        self._total = 0
        self._count = 0
        self.window_size = window_size

    def update(self, sign):
        self._total += 1
        if sign:
            self._count += 1

    def clear(self):
        self._total = 0
        self._count = 0

    def result(self):
        if self._total == self.window_size:
            ratio = self._count / self._total
            self.clear()
            return ratio
