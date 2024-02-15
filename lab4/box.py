from __future__ import annotations
from typing import List
from math import sqrt
import random


class Box:

    @staticmethod
    def find_minimum(func, x0: List[float], explicit: List, implicit: List, alfa=1.3, epsilon=1e-6, trace=False):
        def calculate_xc_xh_xh2(dots: List[List[float]]) -> (List[float], List[float]):
            sorted_list = sorted(dots, key=lambda x: func(*x))
            worst, second_worst = sorted_list[-1], sorted_list[-2]
            suma = [sum(dots[i][j] for i in range(len(dots))) for j in range(len(dots[0]))]
            centroid = [(suma[i] - worst[i]) / (len(dots) - 1) for i in range(len(suma))]
            return centroid, worst, second_worst

        def reflection(xc, xh):
            return [xc[i] + alfa * (xc[i] - xh[i]) for i in range(len(xc))]

        for i in range(len(explicit)):  # check restraints
            if explicit[i][0] > x0[i] or explicit[i][1] < x0[i]:
                print("x0 does not satisfy the initial explicit constraints")
                return
        for i in range(len(implicit)):
            if not implicit[i](*x0):
                print("x0 does not satisfy the initial implicit constraints")
                return

        xc, xh, xh2 = x0, [], []
        n = len(x0)
        points = [x0]
        divergence_counter = 0
        old_minimum = func(*xc)

        for t in range(2 * n):  # generate new points
            x_new = [0 for i in range(n)]
            for i in range(n):
                r = random.uniform(0, 1)  # random from [0, 1]
                x_new[i] = explicit[i][0] + r * (explicit[i][1] - explicit[i][0])

            while True:
                all_satisfied = 1
                for i in range(len(implicit)):
                    if not implicit[i](*x_new):
                        all_satisfied = 0
                        break
                if all_satisfied == 1:
                    break
                x_new = [0.5 * (x_new[i] + xc[i]) for i in range(n)]

            points.append(x_new)
            xc, xh, xh2 = calculate_xc_xh_xh2(points)

        while sqrt(sum((xc[i] - xh[i]) ** 2 for i in range(n))) > epsilon:
            xc, xh, xh2 = calculate_xc_xh_xh2(points)
            xr = reflection(xc, xh)
            for i in range(n):  # explicit restraints
                if xr[i] < explicit[i][0]:
                    xr[i] = explicit[i][0]
                elif xr[i] > explicit[i][1]:
                    xr[i] = explicit[i][1]

            while True:  # implicit restraints
                all_satisfied = 1
                for i in range(len(implicit)):
                    if not implicit[i](*xr):
                        all_satisfied = 0
                        break
                if all_satisfied == 1:
                    break
                xr = [0.5 * (xr[i] + xc[i]) for i in range(n)]

            if func(*xr) > func(*xh2):  # if it is still the worst point
                xr = [0.5 * (xr[i] + xc[i]) for i in range(n)]

            h = points.index(xh)
            points[h] = xr.copy()
            new_minimum = func(*xr)
            if old_minimum <= new_minimum:
                divergence_counter += 1
            else:
                divergence_counter = 0
            old_minimum = new_minimum
            if divergence_counter >= 10:
                return xr

            if trace:
                print("Xr: {}, f(Xr): {:.10f}".format(xr, func(*xr)))

        return sorted(points, key=lambda x: func(*x))[0]
