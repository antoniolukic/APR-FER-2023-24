from .matrix import Matrix


def demo1():
    A = Matrix.read_from_file("demo1.txt")
    A.print_to_terminal("Vrijednost matrice A:", precise=True)
    B = A.scalar(1 / 3)
    B = B.scalar(3)
    B.print_to_terminal("Vrijednost matrice B:", precise=True)
    print("A jednako B: " + str(A == B))


def demo2():
    A = Matrix.read_from_file("demo2.txt")
    b = Matrix(3, 1, [[12], [12], [1]])
    x1 = A.solve(b, True)
    x1.print_to_terminal("LUP dekompozicijom:")
    x2 = A.solve(b, False)
    x2.print_to_terminal("LU dekompozicijom:")


def demo3():
    A = Matrix.read_from_file("demo3.txt")
    b = Matrix(3, 1, [[1], [2], [3]])
    x = A.solve(b, True)  # ovaj sustav nema nikada rješenje, matrica je singularna


def demo4():
    A = Matrix.read_from_file("demo4.txt")
    b = Matrix(3, 1, [[12000000.000001], [14000000], [10000000]])
    x1 = A.solve(b, True)
    x1.print_to_terminal("LUP dekompozicijom:", True)
    x2 = A.solve(b, False)
    x2.print_to_terminal("LU dekompozicijom:", True)


def demo5():
    A = Matrix.read_from_file("demo5.txt")
    b = Matrix(3, 1, [[6], [9], [3]])
    x = A.solve(b, True)
    x.print_to_terminal("LUP dekompozicijom:", True)


def demo6():
    A = Matrix.read_from_file("demo6.txt")
    # A.eps = 1e-9
    A.eps = 1e-6
    b = Matrix(3, 1, [[9000000000], [15], [0.0000000015]])
    x = A.solve(b, True)
    x.print_to_terminal("LUP dekompozicijom:", True)


def demo7():
    A = Matrix.read_from_file("demo3.txt")
    _A = A.inverse()
    _A.print_to_terminal("Inverz matrice:")


def demo8():
    A = Matrix.read_from_file("demo8.txt")
    _A = A.inverse()
    _A.print_to_terminal("Inverz matrice:", True)


def demo9():
    A = Matrix.read_from_file("demo8.txt")
    print("Determinanta matrice: " + str(A.determinant()))


def demo10():
    A = Matrix.read_from_file("demo10.txt")
    print("Determinanta matrice: " + str(A.determinant()))

# demo1()
# demo2()
# demo3()
# demo4()
# demo5()
# demo6()
# demo7()
# demo8()
# demo9()
# demo10()
