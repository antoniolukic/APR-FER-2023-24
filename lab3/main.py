from __future__ import annotations

import math
from typing import List
from gradient_descent import GradientDescent
from newton_raphson import NewtonRaphson
from gauss_newton import GaussNewton


def f_1(x: List[float]):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f_1_derivation(x: List[float]):
    first = 400 * x[0] ** 3 + (2 - 400 * x[1]) * x[0] - 2
    second = 200 * (x[1] - x[0] ** 2)
    return [first, second]


def f_1_hesse(x: List[float]):
    first = -400 * (-3 * x[0] ** 2 + x[1]) + 2
    second = -400 * x[0]
    third = 200
    return [[first, second], [second, third]]


def f_1_G(x: List[float]):
    first = 100 * (x[1] - x[0] ** 2) ** 2
    second = (1 - x[0]) ** 2
    return [first, second]


def f_1_jacob(x: List[float]):
    first = 400 * x[0] * (x[0] ** 2 - x[1])
    second = 200 * (x[1] - x[0] ** 2)
    third = 2 * x[0] - 2
    return [[first, second], [third, 0]]


def f_2(x: List[float]):
    return (x[0] - 4) * (x[0] - 4) + 4 * (x[1] - 2) * (x[1] - 2)


def f_2_derivation(x: List[float]):
    first = 2 * (x[0] - 4)
    second = 8 * (x[1] - 2)
    return [first, second]


def f_2_hesse(x: List[float]):
    return [[2, 0], [0, 8]]


def f_3(x: List[float]):
    return (x[0] - 2) * (x[0] - 2) + (x[1] + 3) * (x[1] + 3)


def f_3_derivation(x: List[float]):
    first = 2 * (x[0] - 2)
    second = 2 * (x[1] + 3)
    return [first, second]


def f_3_hesse(x: List[float]):
    return [[2, 0], [0, 2]]


def f_4(x: List[float]):
    return 1 / 4 * x[0] ** 4 - x[0] ** 2 + 2 * x[0] + (x[1] - 1) * (x[1] - 1)


def f_4_derivation(x: List[float]):
    first = x[0] ** 3 - 2 * x[0] + 2
    second = 2 * (x[1] - 1)
    return [first, second]


def f_4_hesse(x: List[float]):
    first = 3 * x[0] ** 2 - 2
    return [[first, 0], [0, 2]]


def demo1():
    x0 = [0, 0]
    print("Without linear search:")
    print("GradientDescent (x_min, f(x_min), n_derivation_calls, n_function_calls): {}".format(
        GradientDescent.find_minimum(f_3, f_3_derivation, x0, linear_search=False, trace=False)))
    print("With linear search:")
    print("GradientDescent (x_min, f(x_min), n_derivation_calls, n_function_calls): {}".format(
        GradientDescent.find_minimum(f_3, f_3_derivation, x0, linear_search=True, trace=False)))
    # zakljucujemo da gradijentni spust bez linijskog pretraživanja nema globalnu konvergenciju


def demo2():
    print("f1:")
    x0 = [-1.9, 2]
    print("GradientDescent (x_min, f(x_min), n_derivation_calls, n_function_calls): {}".format(
        GradientDescent.find_minimum(f_1, f_1_derivation, x0, linear_search=True, trace=False)))
    print("Newton-Raphson (x_min, f(x_min), n_derivation_calls, n_function_calls, n_hessian_calls): {}".format(
        NewtonRaphson.find_minimum(f_1, f_1_derivation, f_1_hesse, x0, linear_search=True, trace=False)))
    print("f2:")
    x0 = [0.1, 0.3]
    print("GradientDescent (x_min, f(x_min), n_derivation_calls, n_function_calls): {}".format(
        GradientDescent.find_minimum(f_2, f_2_derivation, x0, linear_search=True, trace=False)))
    print("Newton-Raphson (x_min, f(x_min), n_derivation_calls, n_function_calls, n_hessian_calls): {}".format(
        NewtonRaphson.find_minimum(f_2, f_2_derivation, f_2_hesse, x0, linear_search=True, trace=False)))
    # newton-raphson koristi puno manje poziva derivacije i poziva funkcije


def demo3():
    print("Without linear search:")
    x0 = [3, 3]
    print("x0:", x0)
    print("Newton-Raphson (x_min, f(x_min), n_derivation_calls, n_function_calls, n_hessian_calls): {}".format(
        NewtonRaphson.find_minimum(f_4, f_4_derivation, f_4_hesse, x0, linear_search=False, trace=False)))
    x0 = [1, 2]
    print("x0:", x0)
    print("Newton-Raphson (x_min, f(x_min), n_derivation_calls, n_function_calls, n_hessian_calls): {}".format(
        NewtonRaphson.find_minimum(f_4, f_4_derivation, f_4_hesse, x0, linear_search=False, trace=False)))
    print("With linear search:")
    x0 = [3, 3]
    print("x0:", x0)
    print("Newton-Raphson (x_min, f(x_min), n_derivation_calls, n_function_calls, n_hessian_calls): {}".format(
        NewtonRaphson.find_minimum(f_4, f_4_derivation, f_4_hesse, x0, linear_search=True, trace=False)))
    x0 = [1, 2]
    print("x0:", x0)
    print("Newton-Raphson (x_min, f(x_min), n_derivation_calls, n_function_calls, n_hessian_calls): {}".format(
        NewtonRaphson.find_minimum(f_4, f_4_derivation, f_4_hesse, x0, linear_search=True, trace=False)))
    # koristeci linearno pretraživanje iz obje tocke minimum je pronađen dok iz tocke (1, 2)
    # bez linijskog pretraživanje zapelo (overshoot najvjerojatnije)


def demo4():
    def f(x: List[float]):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def G(x: List[float]):
        first = 10 * (x[1] - x[0] ** 2)
        second = (1 - x[0])
        return [first, second]

    def jacob(x: List[float]):
        first = -20 * x[0]
        return [[first, 10], [-1, 0]]

    x0 = [-1.9, 2]
    print("Gauss-Newton (x_min, f(x_min), n_jacobian_calls, n_function_calls, n_g_calls): {}".format(
        GaussNewton.find_minimum(f, G, jacob, x0, linear_search=True, trace=False)))
    # dobivamo precizno rjesenje uz malo poziva


def demo5():
    def f(x: List[float]):
        return (x[0] ** 2 + x[1] ** 2 - 1) ** 2 + (x[1] - x[0] ** 2) ** 2

    def G(x: List[float]):
        first = x[0] ** 2 + x[1] ** 2 - 1
        second = (x[1] - x[0] ** 2)
        return [first, second]

    def jacob(x: List[float]):
        first = 2 * x[0]
        second = 2 * x[1]
        third = -2 * x[0]
        return [[first, second], [third, 1]]

    x0 = [-2, 2]
    print("x0:", x0)
    print("Gauss-Newton (x_min, f(x_min), n_jacobian_calls, n_function_calls, n_g_calls): {}".format(
        GaussNewton.find_minimum(f, G, jacob, x0, linear_search=True, trace=False)))
    x0 = [2, 2]
    print("x0:", x0)
    print("Gauss-Newton (x_min, f(x_min), n_jacobian_calls, n_function_calls, n_g_calls): {}".format(
        GaussNewton.find_minimum(f, G, jacob, x0, linear_search=True, trace=False)))
    x0 = [2, -2]
    print("x0:", x0)
    print("Gauss-Newton (x_min, f(x_min), n_jacobian_calls, n_function_calls, n_g_calls): {}".format(
        GaussNewton.find_minimum(f, G, jacob, x0, linear_search=True, trace=False)))
    # iz prve pocetne tocke dolazimo do jednog globalnog minimuma, iz druge tocke dolazimo do drugog
    # lokalnog minimuma dok iz trece zapinjemo u lokalnom minimumu


def demo6():
    def common_expr(x: List[float]) -> List[float]:
        first = x[0] * math.exp(x[1] * 1) + x[2] - 3
        second = x[0] * math.exp(x[1] * 2) + x[2] - 4
        third = x[0] * math.exp(x[1] * 3) + x[2] - 4
        fourth = x[0] * math.exp(x[1] * 5) + x[2] - 5
        fifth = x[0] * math.exp(x[1] * 6) + x[2] - 6
        sixth = x[0] * math.exp(x[1] * 7) + x[2] - 8
        return [first, second, third, fourth, fifth, sixth]

    def f(x: List[float]) -> float:
        expr = common_expr(x)
        return sum(val ** 2 for val in expr)

    def G(x: List[float]) -> List[float]:
        return common_expr(x)

    def jacob(x: List[float]):
        first = [math.exp(x[1]), x[0] * math.exp(x[1]), 1]
        second = [math.exp(2 * x[1]), 2 * x[0] * math.exp(2 * x[1]), 1]
        third = [math.exp(3 * x[1]), 3 * x[0] * math.exp(3 * x[1]), 1]
        fourth = [math.exp(5 * x[1]), 5 * x[0] * math.exp(5 * x[1]), 1]
        fifth = [math.exp(6 * x[1]), 6 * x[0] * math.exp(6 * x[1]), 1]
        sixth = [math.exp(7 * x[1]), 7 * x[0] * math.exp(7 * x[1]), 1]

        return [first, second, third, fourth, fifth, sixth]

    x0 = [1, 1, 1]
    print("x0:", x0)
    print("Gauss-Newton (x_min, f(x_min), n_jacobian_calls, n_function_calls, n_g_calls): {}".format(
        GaussNewton.find_minimum(f, G, jacob, x0, linear_search=True, trace=False)))


# demo1()
# demo2()
# demo3()
# demo4()
# demo5()
# demo6()
