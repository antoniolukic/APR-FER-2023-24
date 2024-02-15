from __future__ import annotations
import numpy as np
from euler import Euler
from reverese_euler import ReverseEuler
from trapezoid import Trapezoid
from runge_kuta import RungeKutta4
from pece import PECE
import matplotlib.pyplot as plt


def plot_error(error, T, title):
    num_ticks = len(error)
    x_ticks = np.arange(0, num_ticks * T, T)
    plt.plot(x_ticks, error)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Error')
    plt.title("Cumulative Error Over Time " + title)
    plt.show()


def plot_variables(values, T, title):
    x_ticks = []
    x_i = 0
    for i in range(len(values)):
        x_ticks.append(x_i)
        x_i += T
    for i in range(len(values[0])):
        y = [values[j][i][0] for j in range(len(values))]
        plt.plot(x_ticks, y, label="x{}".format(i + 1))
    plt.xlabel('Time (seconds)')
    plt.ylabel('x')
    plt.title("Movement of varibles " + title)
    plt.legend()
    plt.show()


def demo1():
    A = np.array([[0, 1], [-1, 0]])  # za prigušenje stavi A[1][1] na nešto negativno
    B = np.array([[0, 0], [0, 0]])
    T = 0.01
    t_max = 10
    x0 = np.array([[1], [1]])

    def r(t):
        return np.array([[0], [0]])

    def real(t):
        return np.array([[1 * np.cos(t) + 1 * np.sin(t)],
                         [1 * np.cos(t) - 1 * np.sin(t)]])

    methods = [
        ("Euler", Euler),
        ("ReverseEuler", ReverseEuler),
        ("Trapezoid", Trapezoid),
        ("RungeKutta4", RungeKutta4),
        ("PE(CE)^2", (PECE, Euler, ReverseEuler, 2)),
        ("PECE", (PECE, Euler, Trapezoid, 1))
    ]

    for class_name, method_args in methods:
        if isinstance(method_args, tuple):
            class_object = method_args[0]
            method_args = method_args[1:]
            solution, values, cumulative = class_object.determine(A, B, r, x0, T, t_max, 0, real, -1, *method_args)
        else:
            class_object = method_args
            solution, values, cumulative = class_object.determine(A, B, r, x0, T, t_max, 0, real, -1)

        plot_error(cumulative, T, class_name)
        plot_variables(values, T, class_name)
        print(f"{class_name} error:", cumulative[-1])


def demo2():
    A = np.array([[0, 1], [-200, -102]])
    B = np.array([[0, 0], [0, 0]])
    T = 0.01  # moramo smanjiti s 0.1 na 0.01 zbog stabilnosti
    t_max = 1
    x0 = np.array([[1], [-2]])

    def r(t):
        return np.array([[0], [0]])

    def real(t):
        return 0

    solution, values, cumulative = Euler.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "Euler")
    solution, values, cumulative = ReverseEuler.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "ReverseEuler")
    solution, values, cumulative = Trapezoid.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "Trapezoid")
    solution, values, cumulative = RungeKutta4.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "RungeKutta4")
    solution, values, cumulative = PECE.determine(A, B, r, x0, T, t_max, 0, real, -1, Euler, ReverseEuler, 2)
    plot_variables(values, T, "PE(CE)^2")
    solution, values, cumulative = PECE.determine(A, B, r, x0, T, t_max, 0, real, -1, Euler, Trapezoid, 1)
    plot_variables(values, T, "PECE")


def demo3():
    A = np.array([[0, -2], [1, -3]])
    B = np.array([[2, 0], [0, 3]])
    T = 0.01
    t_max = 10
    x0 = np.array([[1], [3]])

    def r(t):
        return np.array([[1], [1]])

    def real(t):
        return 0

    solution, values, cumulative = Euler.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "Euler")
    solution, values, cumulative = ReverseEuler.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "ReverseEuler")
    solution, values, cumulative = Trapezoid.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "Trapezoid")
    solution, values, cumulative = RungeKutta4.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "RungeKutta4")
    solution, values, cumulative = PECE.determine(A, B, r, x0, T, t_max, 0, real, -1, Euler, ReverseEuler, 2)
    plot_variables(values, T, "PE(CE)^2")
    solution, values, cumulative = PECE.determine(A, B, r, x0, T, t_max, 0, real, -1, Euler, Trapezoid, 1)
    plot_variables(values, T, "PECE")


def demo4():
    A = np.array([[1, -5], [1, -7]])
    B = np.array([[5, 0], [0, 3]])
    T = 0.01
    t_max = 1
    x0 = np.array([[-1], [3]])

    def r(t):
        return np.array([[t], [t]])

    def real(t):
        return 0

    solution, values, cumulative = Euler.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "Euler")
    solution, values, cumulative = ReverseEuler.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "ReverseEuler")
    solution, values, cumulative = Trapezoid.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "Trapezoid")
    solution, values, cumulative = RungeKutta4.determine(A, B, r, x0, T, t_max, 0, real, -1)
    plot_variables(values, T, "RungeKutta4")
    solution, values, cumulative = PECE.determine(A, B, r, x0, T, t_max, 0, real, -1, Euler, ReverseEuler, 2)
    plot_variables(values, T, "PE(CE)^2")
    solution, values, cumulative = PECE.determine(A, B, r, x0, T, t_max, 0, real, -1, Euler, Trapezoid, 1)
    plot_variables(values, T, "PECE")


# demo1()
# demo2()
# demo3()
# demo4()
