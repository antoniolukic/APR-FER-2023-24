from __future__ import annotations
from typing import List
from math import sqrt, log
from lab2.hooke_jeeves import HookJeeves


class InnerPointFinder:
    def inner_point(self, g: List, t: float, x0: List[float], epsilon=1e-6):
        x = x0.copy()
        while True:
            xs = x.copy()

            def redirect(x):
                return self.G(g, x)

            x, counter = HookJeeves.find_minimum(x, redirect)
            if sqrt(sum((xs[i] - x[i]) ** 2 for i in range(len(xs)))) < epsilon or self.G(g, x) == 0:
                break
        return x

    def G(self, g: List, x: List[float]) -> float:
        sol = 0
        for gi in g:
            value = gi(*x)
            if value < 0:
                sol -= value
        return sol


class MixedTransformation:
    def minimum_mixed(self, func, g: List, h: List, t0: float, x0: List[float], epsilon=1e-6):
        finder = InnerPointFinder()
        x = finder.inner_point(g, t0, x0)
        print("Point that satisfies the constraints x0: {}".format(x))
        t = t0
        old_minimum = self.F(func, g, h, t, x)
        divergence_counter = 0

        while True:
            xs = x.copy()

            def redirect(x):
                return self.F(func, g, h, t, x)

            x, counter = HookJeeves.find_minimum(x, redirect)
            new_minimum = self.F(func, g, h, t, x)
            if old_minimum > new_minimum:
                divergence_counter += 1
            else:
                divergence_counter = 0
            old_minimum = new_minimum
            if divergence_counter >= 10:
                return x
            t *= 10
            if sqrt(sum((xs[i] - x[i]) ** 2 for i in range(len(xs)))) < epsilon:
                break
        return x

    def F(self, func, g: List, h: List, t, x):
        sol = func(*x)
        for gi in g:
            value = gi(*x)
            if value <= 0:
                return float('inf')
            else:
                sol -= 1 / t * log(value)

        for hi in h:
            sol += t * (hi(*x) ** 2)
        return sol
