from __future__ import annotations
from .algorithm import Algorithm


class HookJeeves(Algorithm):

    @staticmethod
    def find_minimum(x_0, func, dx=None, epsilon=None, trace=False):
        # x_0 is now a vector
        if dx is None:
            dx = [0.5] * len(x_0)
        if epsilon is None:
            epsilon = [1e-6] * len(x_0)

        def explore(x_p, dx):
            x = x_p.copy()
            counter = 0

            for i in range(len(x_p)):
                P = func(x)
                x[i] += dx[i]
                N = func(x)
                counter += 1
                if N > P:
                    x[i] -= 2 * dx[i]
                    N = func(x)
                    counter += 1
                    if N > P:
                        x[i] += dx[i]
            return x, counter

        x_b, x_p = x_0.copy(), x_0.copy()
        counter = 0
        while True:
            x_n, curr_counter = explore(x_p, dx)
            counter += curr_counter

            if trace:
                for name, value in zip(iter(['x_b', 'x_p', 'x_n']), iter([x_b, x_p, x_n])):
                    print("{}={}".format(name, '(' + ', '.join('{:>8.6f}'.format(item) for item in value) + ')'), end='')
                    print(", f({})={} | ".format(name, func(value)), end='')
                print()

            if func(x_n) < func(x_b):
                x_p = [2 * x_ni - x_bi for x_ni, x_bi in zip(x_n, x_b)]
                x_b = x_n.copy()
            else:
                dx = [dx_i / 2 for dx_i in dx]
                x_p = x_b.copy()

            if max(dx) <= max(epsilon):
                return x_b, counter
