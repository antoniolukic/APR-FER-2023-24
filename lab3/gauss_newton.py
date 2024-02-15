from __future__ import annotations
from typing import List
from math import sqrt
from lab2.golden_cut import GoldenCut
from lab1.matrix import Matrix


class GaussNewton:

    @staticmethod
    def find_minimum(function, G_function, jacobian, x0: List[float], epsilon=1e-6, linear_search=False, trace=False) \
            -> (List[float], float, int, int, int):

        jacob = jacobian(x0)
        jacob_matrix = Matrix(len(jacob), len(jacob[0]), jacob)
        G_ = [G_function(x0)]
        G_matrix = Matrix(len(G_), len(G_[0]), G_).transpose()

        old_minimum = function(x0)
        jacobian_counter = 1
        g_counter = 1
        function_counter = 1
        divergence_counter = 0

        magnitude = 10e5

        while magnitude > epsilon:
            A = jacob_matrix.transpose() * jacob_matrix
            g = jacob_matrix.transpose() * G_matrix
            delta = A.solve(g.scalar(-1))

            if linear_search:
                def redirect(lamb):
                    shifted = x0.copy()
                    for i in range(len(shifted)):
                        shifted[i] = shifted[i] + lamb[0] * delta.get(i + 1, 1)
                    return function(shifted)

                step_size, counter = GoldenCut.find_minimum([0], None, redirect)
                function_counter += counter
            else:
                step_size = -1

            x_next = [x0[i] + step_size * delta.get(i + 1, 1) for i in range(len(x0))]

            new_minimum = function(x_next)
            function_counter += 1
            if old_minimum <= new_minimum:
                divergence_counter += 1
            else:
                divergence_counter = 0
            old_minimum = new_minimum

            if trace:
                print("A:")
                A.print_to_terminal()
                print("g:")
                g.print_to_terminal()
                x_next_str = ", ".join("{:.10f}".format(x) for x in x_next)
                print("function: {:.10f}, lambda: {:.10f}, x_next: [{}]".format(function(x0), step_size, x_next_str))

            if divergence_counter >= 10:
                print("The gauss-newton method diverged or local maxima or cannot move lower because of step size.")
                return x0, function(x0), jacobian_counter, function_counter, g_counter

            x0 = x_next
            jacob = jacobian(x0)
            jacob_matrix = Matrix(len(jacob), len(jacob[0]), jacob)
            jacobian_counter += 1
            G_ = [G_function(x0)]
            G_matrix = Matrix(len(G_), len(G_[0]), G_).transpose()
            g_counter += 1

            magnitude = sqrt(sum([delta.get(i + 1, 1) * delta.get(i + 1, 1) * abs(step_size) for i in range(len(x0))]))

        return x0, function(x0), jacobian_counter, function_counter, g_counter
