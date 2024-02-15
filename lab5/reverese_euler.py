from __future__ import annotations
import numpy as np


class ReverseEuler:

    @staticmethod
    def determine(A: np.ndarray, B: np.ndarray, r, x0: np.ndarray, T: float, t_max: float, t0: float, real,
                  n_trace: int, x1: np.ndarray = None, implicit=False):
        values, cumulative = [x0], [0]
        t, i = t0, 0

        while t <= t_max:
            if implicit and x1 is not None:
                x_new = x0 + T * (A @ x1 + B @ r(t + T))
            else:
                x_new = np.linalg.inv(np.identity(len(A)) - T * A) @ (x0 + T * B @ r(t))

            if n_trace > 0 and i % n_trace == 0:
                print(x_new)

            values.append(x_new)
            cumulative.append(np.sum(np.abs(x_new - real(t + T))) + cumulative[i])
            x0 = x_new

            t += T
            i += 1

        return x0, values, cumulative
