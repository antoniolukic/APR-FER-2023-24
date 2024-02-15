from __future__ import annotations
from typing import List


class UnimodalInterval:

    @staticmethod
    def find_interval(x_0: List[float], func, h: int) -> (float, float, int):
        # x_0 has only 1 dimension
        x_0 = x_0[0]
        f_minus, f_x0, f_plus = func([x_0 - h]), func([x_0]), func([x_0 + h])
        if f_plus > f_x0 and f_minus > f_x0:
            return x_0 - h, x_0 + h, 3

        sign = -1 if f_plus > f_minus else 1
        h *= 2

        previous = f_x0
        new = func([x_0 + sign * h])
        counter = 4
        while new < previous:
            h *= 2
            previous = new
            counter += 1

        left = min(x_0 + sign * h, x_0 + sign * h / 4)
        right = max(x_0 + sign * h, x_0 + sign * h / 4)
        return left, right, counter
