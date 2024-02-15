from __future__ import annotations
import numpy as np
from time import sleep


class PECE:
    @staticmethod
    def determine(A: np.ndarray, B: np.ndarray, r, x0: np.ndarray, T: float, t_max: float, t0: float, real, n_trace: int,
                  predictor, corrector, c_times: int):
        values, cumulative = [x0], [0]
        t, i = t0, 0

        while t <= t_max:
            x_pred, _, __ = predictor.determine(A, B, r, x0, T, t, t, real, -1)
            x_corr = x_pred
            for j in range(c_times):
                x_corr, _, __ = corrector.determine(A, B, r, x0, T, t, t, real, -1, x_corr, True)

            x_new = x_corr

            if n_trace > 0 and i % n_trace == 0:
                print(x_new)

            values.append(x_new)
            cumulative.append(np.sum(np.abs(x_new - real(t + T))) + cumulative[i])
            x0 = x_new

            t += T
            i += 1

        return x0, values, cumulative
