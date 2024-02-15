from __future__ import annotations
import numpy as np


class RungeKutta4:
    @staticmethod
    def determine(A: np.ndarray, B: np.ndarray, r, x0: np.ndarray, T: float, t_max: float, t0: float, real, n_trace: int):
        values, cumulative = [x0], [0]
        t, i = t0, 0

        while t <= t_max:
            k1 = T * (A @ x0 + B @ r(t))
            k2 = T * (A @ (x0 + 0.5 * k1) + B @ r(t + 0.5 * T))
            k3 = T * (A @ (x0 + 0.5 * k2) + B @ r(t + 0.5 * T))
            k4 = T * (A @ (x0 + k3) + B @ r(t + T))

            x_new = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            if n_trace > 0 and i % n_trace == 0:
                print(x_new)

            values.append(x_new)
            cumulative.append(np.sum(np.abs(x_new - real(t + T))) + cumulative[i])
            x0 = x_new

            t += T
            i += 1

        return x0, values, cumulative
