from __future__ import annotations
from .unimodal_interval import UnimodalInterval
from math import sqrt
from .algorithm import Algorithm


class GoldenCut(Algorithm):

    @staticmethod
    def find_minimum(x_0, interval: (float, float), func, epsilon=1e-6, trace=False):
        # x_0 has only 1 dimension
        k = 0.5 * (sqrt(5) - 1)
        counter = 0

        if interval is None:
            left, right, counter = UnimodalInterval.find_interval(x_0, func, 1)
            interval = (left, right)

        a, b = interval[0], interval[1]
        c = b - k * (b - a)
        d = a + k * (b - a)
        f_c = func([c])
        f_d = func([d])
        counter += 2

        while b - a > epsilon:
            if trace:
                print("a={:>8.6f}, c={:>8.6f}, d={:>8.6f}, b={:>8.6f}".format(a, c, d, b))

            if f_c < f_d:
                b = d
                d = c
                c = b - k * (b - a)
                f_d = f_c
                f_c = func([c])
            else:
                a = c
                c = d
                d = a + k * (b - a)
                f_c = f_d
                f_d = func([d])

            counter += 1

        return (a + b) / 2, counter
