from __future__ import annotations
from typing import List
from math import sqrt
from lab2.golden_cut import GoldenCut
from lab1.matrix import Matrix


class NewtonRaphson:

    @staticmethod
    def find_minimum(function, derivation, hessian, x0: List[float], epsilon=1e-6, linear_search=False, trace=False) \
            -> (List[float], float, int, int, int):

        hess = hessian(x0)
        hess_matrix = Matrix(len(hess), len(hess[0]), hess)
        inverse_hess_matrix = hess_matrix.inverse()
        grad = [derivation(x0)]
        grad_matrix = Matrix(len(grad), len(grad[0]), grad).transpose()

        old_minimum = function(x0)
        divergence_counter = 0
        derivation_counter = 1
        function_counter = 1
        hessian_counter = 1

        magnitude = sqrt(sum([xi * xi for xi in grad[0]]))

        while magnitude > epsilon:
            mul = inverse_hess_matrix * grad_matrix
            if linear_search:
                def redirect(lamb):
                    shifted = x0.copy()
                    for i in range(len(shifted)):
                        shifted[i] = shifted[i] + lamb[0] * mul.get(i + 1, 1)
                    return function(shifted)

                step_size, counter = GoldenCut.find_minimum([0], None, redirect)
                function_counter += counter
            else:
                step_size = -1

            x_next = [x0[i] + step_size * mul.get(i + 1, 1) for i in range(len(x0))]

            new_minimum = function(x_next)
            function_counter += 1
            if old_minimum <= new_minimum:
                divergence_counter += 1
            #  else:
            #    divergence_counter = 0
            old_minimum = new_minimum

            if trace:
                gradient_str = ", ".join("{:.10f}".format(g) for g in grad[0])
                x_next_str = ", ".join("{:.10f}".format(x) for x in x_next)
                print("function: {:.10f}, gradient: [{}], lambda: {:.10f}, x_next: [{}]".format(function(x0),
                                                                                                gradient_str,
                                                                                                step_size, x_next_str))
            if divergence_counter >= 10:
                print("The newton-raphson method diverged or local maxima or cannot move lower because of step size.")
                return x0, function(x0), derivation_counter, function_counter, hessian_counter

            x0 = x_next
            hess = hessian(x0)
            hessian_counter += 1
            hess_matrix = Matrix(len(hess), len(hess[0]), hess)
            inverse_hess_matrix = hess_matrix.inverse()
            grad = [derivation(x0)]
            derivation_counter += 1
            grad_matrix = Matrix(len(grad), len(grad[0]), grad).transpose()
            magnitude = sqrt(sum([xi * xi for xi in grad[0]]))

        return x0, function(x0), derivation_counter, function_counter, hessian_counter
