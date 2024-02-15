from __future__ import annotations
from .golden_cut import GoldenCut
from .coordinate_descent import CoordinateDescent
from .simplex import Simplex
from .hooke_jeeves import HookJeeves
from typing import List
from math import *
import random


def f_0(x: List[float]):
    return (x[0] - 3) ** 2


def f_1(x: List[float]):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def f_2(x: List[float]):
    return (x[0] - 4) ** 2 + 4 * (x[1] - 2) ** 2


def f_3(x: List[float]):
    return sum((x[i - 1] - i) ** 2 for i in range(1, len(x) + 1))


def f_4(x: List[float]):
    return abs((x[0] - x[1]) * (x[0] + x[1])) + sqrt(x[0] ** 2 + x[1] ** 2)


def f_5(x: List[float]):
    suma = sum(x_i ** 2 for x_i in x)
    return 0.5 + (sin(sqrt(suma)) ** 2 - 0.5) / ((1 + 0.001 * suma) ** 2)


def demo1():
    x_0 = [3]
    print("GoldenCut         (x_min, n_of_calls): {}".format(GoldenCut.find_minimum(x_0, None, f_0)))
    print("CoordinateDescent (x_min, n_of_calls): {}".format(CoordinateDescent.find_minimum(x_0, f_0)))
    print("Simplex           (x_min, n_of_calls): {}".format(Simplex.find_minimum(x_0, f_0)))
    print("HookeJeeves       (x_min, n_of_calls): {}".format(HookJeeves.find_minimum(x_0, f_0)))

    # golden cut ostaje gotovo isti, coordinate descnet raste jako brzo dok preostala dva rastu puno sporije


def demo2():
    x_0 = [-1.9, 2]
    print("f1:")
    print("CoordinateDescent (x_min, n_of_calls): {}".format(CoordinateDescent.find_minimum(x_0, f_1)))
    print("Simplex           (x_min, n_of_calls): {}".format(Simplex.find_minimum(x_0, f_1)))
    print("HookeJeeves       (x_min, n_of_calls): {}".format(HookJeeves.find_minimum(x_0, f_1)))
    x_0 = [0.1, 0.3]
    print("f2:")
    print("CoordinateDescent (x_min, n_of_calls): {}".format(CoordinateDescent.find_minimum(x_0, f_2)))
    print("Simplex           (x_min, n_of_calls): {}".format(Simplex.find_minimum(x_0, f_2)))
    print("HookeJeeves       (x_min, n_of_calls): {}".format(HookJeeves.find_minimum(x_0, f_2)))
    x_0 = [0, 0, 0, 0, 0]  # zanimljivo, simplex zaglibi ako je x_0 = [0, 0]
    print("f3:")
    print("CoordinateDescent (x_min, n_of_calls): {}".format(CoordinateDescent.find_minimum(x_0, f_3)))
    print("Simplex           (x_min, n_of_calls): {}".format(Simplex.find_minimum(x_0, f_3)))
    print("HookeJeeves       (x_min, n_of_calls): {}".format(HookJeeves.find_minimum(x_0, f_3)))
    x_0 = [5.1, 1.1]
    print("f4:")  # metode nabolje rješenje nalaze po redu: simplex, kordinatni spust, hookejeeves se može malo micati
    print("CoordinateDescent (x_min, n_of_calls): {}".format(CoordinateDescent.find_minimum(x_0, f_4)))
    print("Simplex           (x_min, n_of_calls): {}".format(Simplex.find_minimum(x_0, f_4)))
    print("HookeJeeves       (x_min, n_of_calls): {}".format(HookJeeves.find_minimum(x_0, f_4)))


def demo3():
    x_0 = [5, 5]  # simplex je sjajno namješten da pronađe mnimum, hookejeeves ne mrda jer mu je odmah svugdje lošije
    print("Simplex           (x_min, n_of_calls): {}".format(Simplex.find_minimum(x_0, f_4)))
    print("HookeJeeves       (x_min, n_of_calls): {}".format(HookJeeves.find_minimum(x_0, f_4)))


def demo4():
    # u slučajevima za step 2, 3, 6, 14, 15 dogadja se da je jedna točka simplexa na drugoj točci
    x_0 = [0.5, 0.5]
    print("x_0: {}".format(x_0))
    for i in range(1, 21):
        print("Simplex move={}   (x_min, n_of_calls): {}".format(i, Simplex.find_minimum(x_0, f_1, move=i)))

    x_0 = [20, 20]
    print("x_0: {}".format(x_0))
    for i in range(1, 21):
        print("Simplex move={}   (x_min, n_of_calls): {}".format(i, Simplex.find_minimum(x_0, f_1, move=i)))


def demo5():
    p = 0
    n = 50
    for i in range(n):
        # rješenje će pronaći jedino ako su komponente početne tčcke je unutar intervala [0.5 0.5]
        x_0 = [random.uniform(-50, 50), random.uniform(-50, 50)]
        x_min, n_of_calls = HookJeeves.find_minimum(x_0, f_5)
        print("Simplex           (x_min, n_of_calls): {} {}".format(x_min, n_of_calls))
        value = f_5(x_min)
        print("value: {}".format(value))
        if value < 1e-4:
            p += 1
    p /= n
    print("Vjerojatnost pronalaženja minimuma je: {}".format(p))


# demo1()
# demo2()
# demo3()
# demo4()
# demo5()
