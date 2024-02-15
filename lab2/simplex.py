from __future__ import annotations
from .algorithm import Algorithm


class Simplex(Algorithm):

    @staticmethod
    def find_minimum(x_0, func, alfa=1, beta=0.5, gamma=2, sigma=0.5, epsilon=1e-6, trace=False, move=1):
        def order_indices(FX):
            return sorted(range(len(FX)), key=lambda x: FX[x])

        def calculate_centroid(X, indices):
            return [sum(X[i][j] for i in indices[:-1]) / len(indices[:-1]) for j in range(len(X[0]))]

        def reflection(X, h, X_c):
            return [X_ci + alfa * (X_ci - X[h][i]) for i, X_ci in enumerate(X_c)]

        def expansion(X_r, X_c):
            return [X_ci + gamma * (X_ri - X_ci) for X_ci, X_ri in zip(X_c, X_r)]

        def contraction(X, h, X_c):
            return [X_ci + beta * (X[h][i] - X_ci) for i, X_ci in enumerate(X_c)]

        # x_0 is now a vector
        n = len(x_0)
        X = [x_0.copy() for x in range(n + 1)]
        for i in range(n):
            X[i + 1][i] += move

        FX = [func(point) for point in X]
        counter = n + 1

        while True:
            indices = order_indices(FX)
            h, l = indices[-1], indices[0]
            X_old = X[h]
            X_c = calculate_centroid(X, indices)

            X_r = reflection(X, h, X_c)
            func_X_r = func(X_r)
            counter += 1

            if func_X_r < FX[l]:
                X_e = expansion(X_r, X_c)
                func_X_e = func(X_e)
                counter += 1

                if func_X_e < FX[l]:
                    X[h] = X_e
                    FX[h] = func_X_e
                else:
                    X[h] = X_r
                    FX[h] = func_X_r
            else:
                if all(func_X_r > FX[j] for j in range(n + 1) if j != h):
                    if func_X_r < FX[h]:
                        X[h] = X_r
                        FX[h] = func_X_r

                    X_k = contraction(X, h, X_c)
                    func_X_k = func(X_k)
                    counter += 1

                    if func_X_k < FX[h]:
                        X[h] = X_k
                        FX[h] = func_X_k
                    else:
                        for i in range(1, n + 1):
                            X[i] = [sigma * (X[l][j] + X[i][j]) for j in range(len(X[0]))]
                            FX[i] = func(X[i])
                            counter += 1
                else:
                    X[h] = X_r
                    FX[h] = func_X_r

            if trace:
                formatted_list = '[' + ', '.join('{:>8.6f}'.format(item) for item in X_c) + ']'
                print("x: " + formatted_list + ", " + str(func(X_c)))

            if max(abs(X[h][i] - X_old[i]) for i in range(len(X[0]))) <= epsilon:
                return X[l], counter
