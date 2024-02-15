from __future__ import annotations
from .algorithm import Algorithm
from .golden_cut import GoldenCut


class CoordinateDescent(Algorithm):

    @staticmethod
    def find_minimum(x_0, func, epsilon=1e-6, trace=False):
        # x_0 is now a vector
        x_move, global_counter = x_0, 0
        dim = len(x_0)

        while True:
            for i in range(dim):
                def redirect(x):
                    whole = x_move[:i] + x + x_move[i + 1:]
                    return func(whole)
                x_best, counter = GoldenCut.find_minimum([x_move[i]], None, redirect)

                x_move = x_move[:i] + [x_best] + x_move[i+1:]
                global_counter += counter

            result = max(abs(x1 - x2) for x1, x2 in zip(x_0, x_move))

            if trace:
                formatted_list = '[' + ', '.join('{:>8.6f}'.format(item) for item in x_move) + ']'
                print("x: " + formatted_list)

            if result <= epsilon:
                return x_move, global_counter

            x_0 = x_move
