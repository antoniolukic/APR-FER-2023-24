from __future__ import annotations
from box import Box
from transform_without_restraints import MixedTransformation


def f1(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def f2(x1, x2):
    return (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2


def f3(x1, x2):
    return (x1 - 2) ** 2 + (x2 + 3) ** 2


def f4(x1, x2):
    return (x1 - 3) ** 2 + x2 ** 2


def demo1():
    explicit = [[-100, 100],
                [-100, 100]]

    def implicit1(x1, x2):
        return x2 - x1 >= 0

    def implicit2(x1, x2):
        return 2 - x1 >= 0

    implicit = [implicit1, implicit2]
    print("f1:")
    point = Box.find_minimum(f1, [-1.9, 2], explicit, implicit, alfa=1.3, trace=False)  # depending on random
    print("x: {}, f(x): {:.10f}".format(point, f1(*point)))
    print("f2:")
    point = Box.find_minimum(f2, [0.1, 0.3], explicit, implicit, alfa=1.3, trace=False)
    print("x: {}, f(x): {:.10f}".format(point, f2(*point)))
    # za f1 se minimum ne mijenja s obzirom na ograničenja, dok se za f2 mijenja


def demo2():
    mixed = MixedTransformation()

    def g1(x1, x2):
        return x2 - x1

    def g2(x1, x2):
        return 2 - x1

    g = [g1, g2]
    print("f1:")
    #  point = mixed.minimum_mixed(f1, g, [], 0.1, [-1.9, 2])
    point = mixed.minimum_mixed(f1, g, [], 0.1, [0.5, 2])
    print("x: {}, f(x): {:.10f}".format(point, f1(*point)))
    print("f2:")
    point = mixed.minimum_mixed(f2, g, [], 0.1, [0.1, 0.3])
    print("x: {}, f(x): {:.10f}".format(point, f2(*point)))
    # za f1 iz poč točke [-1.9, 2] ne dolazi do minimuma dok za početnu točku [0.5, 2] doalzi do minimuma
    # za f2 dolazi do ograničenog minumuma iz zadane poč točke


def demo3():
    mixed = MixedTransformation()

    def g1(x1, x2):
        return 3 - x1 - x2

    def g2(x1, x2):
        return 3 + 1.5 * x1 - x2

    def h1(x1, x2):
        return x2 - 1

    g = [g1, g2]
    h = [h1]
    print("f4:")
    point = mixed.minimum_mixed(f4, g, h, 0.1, [5, 5])
    print("x: {}, f(x): {:.10f}".format(point, f4(*point)))


# demo1()
# demo2()
# demo3()
