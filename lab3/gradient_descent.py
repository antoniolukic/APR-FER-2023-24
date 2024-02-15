from __future__ import annotations
from typing import List
from math import sqrt
from lab2.golden_cut import GoldenCut


class GradientDescent:

    @staticmethod
    def find_minimum(function, derivation, x0: List[float], epsilon=1e-6, linear_search=False, trace=False) -> (
            List[float], float, int, int):

        gradient = derivation(x0)
        old_minimum = function(x0)
        divergence_counter = 0
        derivation_counter = 1
        function_counter = 1

        magnitude = sqrt(sum([xi * xi for xi in gradient]))

        while magnitude > epsilon:
            if linear_search:
                def redirect(lamb):
                    shifted = x0.copy()
                    for i in range(len(shifted)):
                        shifted[i] = shifted[i] + lamb[0] * gradient[i]
                    return function(shifted)

                step_size, counter = GoldenCut.find_minimum([0], None, redirect)
                function_counter += counter
            else:
                step_size = -1

            x_next = [x0_i + step_size * gradient_i for x0_i, gradient_i in zip(x0, gradient)]

            new_minimum = function(x_next)
            function_counter += 1
            if old_minimum <= new_minimum:
                divergence_counter += 1
            else:
                divergence_counter = 0
            old_minimum = new_minimum

            if trace:
                gradient_str = ", ".join("{:.10f}".format(g) for g in gradient)
                x_next_str = ", ".join("{:.10f}".format(x) for x in x_next)
                print("function: {:.10f}, gradient: [{}], lambda: {:.10f}, x_next: [{}]".format(function(x0),
                                                                                                gradient_str,
                                                                                                step_size, x_next_str))
            if divergence_counter >= 10:  # gradient descent with linear search cannot diverge
                print("The gradient descent method diverged or local maxima or cannot move lower because of step size.")
                return x0, function(x0), derivation_counter, function_counter

            x0 = x_next
            gradient = derivation(x0)
            derivation_counter += 1
            magnitude = sqrt(sum([xi * xi for xi in gradient]))

        return x0, function(x0), derivation_counter, function_counter
