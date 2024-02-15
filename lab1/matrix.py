from __future__ import annotations
import re
import copy


class Matrix:
    eps = 1e-9

    def __init__(self, n, m, values):
        self.n = n
        self.m = m
        self.values = values

    def check(self, i, j):
        return 1 <= i <= self.n and 1 <= j <= self.m

    def set(self, i, j, value):
        if not self.check(i, j):
            raise Exception("invalid indexes")
        self.values[i - 1][j - 1] = value

    def get(self, i, j) -> float:
        if not self.check(i, j):
            raise Exception("invalid indexes")
        return self.values[i - 1][j - 1]

    def __add__(self, other: Matrix):
        if self.n != other.n or self.m != other.m:
            raise Exception("Invalid matrix dimensions")

        new_values = [[self.values[j][i] + other.values[j][i] for i in range(self.m)] for j in range(self.n)]

        return Matrix(self.n, self.m, new_values)

    def __iadd__(self, other: Matrix):
        return self.__add__(other)

    def __sub__(self, other: Matrix):
        if self.n != other.n or self.m != other.m:
            raise Exception("Invalid matrix dimensions")

        new_values = [[self.values[j][i] - other.values[j][i] for i in range(self.m)] for j in range(self.n)]

        return Matrix(self.n, self.m, new_values)

    def __isub__(self, other: Matrix):
        return self.__isub__(other)

    def __mul__(self, other: Matrix):
        if self.m != other.n:
            raise Exception("matrix dimensions are not compatible for multiplication")

        n = self.n
        m = other.m
        result = [[0] * m for _ in range(n)]

        for i in range(n):
            for j in range(m):
                for k in range(self.m):
                    result[i][j] += self.values[i][k] * other.values[k][j]

        return Matrix(n, m, result)

    def __eq__(self, other) -> bool:
        if self.n != other.n or self.m != other.m:
            return True

        for i1, i2 in zip(self.values, other.values):
            for j1, j2 in zip(i1, i2):
                if j1 != j2:
                    return False
        return True

    def transpose(self):
        n = len(self.values)
        m = len(self.values[0])

        transpose = [[self.values[j][i] for j in range(n)] for i in range(m)]

        return Matrix(m, n, transpose)

    def scalar(self, value):
        new_values = [[j * value for j in i] for i in self.values]
        return Matrix(self.n, self.m, new_values)

    @staticmethod
    def forward_substitution(matrix: Matrix, b: Matrix):
        if b.n != matrix.n:
            raise Exception("invalid free vector dimensions")
        x = [0.0] * matrix.n
        for i in range(matrix.n):
            x[i] = b.get(i + 1, 1)
            for j in range(i):
                x[i] -= matrix.values[i][j] * x[j]
        return Matrix(matrix.n, 1, [[i] for i in x])

    @staticmethod
    def backward_substitution(matrix: Matrix, b: Matrix):
        if b.n != matrix.n:
            raise Exception("invalid free vector dimensions")
        x = [0.0] * matrix.n
        for i in range(matrix.n - 1, -1, - 1):
            x[i] = b.get(i + 1, 1)
            for j in range(i + 1, matrix.n):
                x[i] -= matrix.values[i][j] * x[j]
            x[i] /= matrix.values[i][i]
        return Matrix(matrix.n, 1, [[i] for i in x])

    @staticmethod
    def max_in_column(matrix: Matrix, start: int):
        curr = abs(matrix.values[start][start])
        where = start
        for i in range(start + 1, matrix.n):
            if abs(matrix.values[i][start]) > curr:
                curr = abs(matrix.values[i][start])
                where = i
        return where, curr

    @staticmethod
    def swap_rows(matrix: Matrix, first, second):
        matrix.values[first], matrix.values[second] = matrix.values[second], matrix.values[first]

    def lu_decomposition(self, pivot=True):
        decomposed = copy.deepcopy(self)
        perm_vec = [i for i in range(self.n)]
        n_swaps = 1
        for i in range(self.n):
            if pivot:
                largest, value = Matrix.max_in_column(decomposed, i)
                if i != largest:
                    n_swaps *= -1

                    Matrix.swap_rows(decomposed, i, largest)
                    perm_vec[i], perm_vec[largest] = perm_vec[largest], perm_vec[i]

            if abs(decomposed.values[i][i]) <= self.eps:
                raise Exception("matrix is singular")

            for j in range(i + 1, self.n):
                factor = decomposed.values[j][i] / decomposed.values[i][i]
                for k in range(i + 1, self.n):
                    decomposed.values[j][k] -= factor * decomposed.values[i][k]
                decomposed.values[j][i] = factor

        return decomposed, perm_vec, n_swaps

    def determinant(self):
        lu, perm_vec, n_swaps = self.lu_decomposition(True)
        det = n_swaps
        for i in range(lu.n):
            det *= lu.get(i + 1, i + 1)
        return det

    def inverse(self):
        lu, perm_vec, n_swaps = self.lu_decomposition(True)
        P = [[0] * lu.n for _ in range(lu.m)]
        for i in range(len(perm_vec)):
            P[i][perm_vec[i]] = 1

        inverse = []
        for i in range(lu.n):
            b = Matrix(lu.n, 1, [[row[i]] for row in P])
            y = Matrix.forward_substitution(lu, b)
            x = Matrix.backward_substitution(lu, y)
            inverse.append([xi[0] for xi in x.values])
        inverse = Matrix(lu.n, lu.m, inverse)
        return inverse.transpose()

    def solve(self, b: Matrix, pivot=True):
        lu, perm_vec, n_swaps = self.lu_decomposition(pivot)
        b.values = [b.values[i] for i in perm_vec]
        y = Matrix.forward_substitution(lu, b)
        x = Matrix.backward_substitution(lu, y)
        return x

    def print_to_terminal(self, text=None, precise=False):
        if text is not None:
            print(text)
        for i in range(self.n):
            for j in range(self.m):
                if precise:
                    print("{:>30.20f}".format(self.values[i][j]), end=" ")
                else:
                    print("{:>10.5f}".format(self.values[i][j]), end=" ")
            print()
        print()

    def print_to_file(self, path):
        f = open(path, "w")
        curr = ""
        for i in self.values:
            for j in i:
                curr += str(j) + " "
            curr = curr[:-1] + "\n"
        f.write(curr)
        f.close()

    @staticmethod
    def read_from_file(path):
        values = []
        f = open(path, "r")
        line = ' '
        while line != '\n' and line != '':
            line = f.readline()
            if len(re.split(r'\s+', line)[:-1]) > 0:
                values.append([float(x) for x in re.split(r'\s+', line)[:-1]])
        f.close()

        return Matrix(len(values), len(values[0]), values)


#A = Matrix.read_from_file("demo1.txt")
#A.print_to_terminal("Matrica A:")
#LU, perm_vec, n_swaps = A.lu_decomposition(True)
#LU.print_to_terminal("LU dekompozicija matrice A:")
#print("Determinanta matrice A:\n" + str(A.determinant()))
#A.inverse().print_to_terminal("Inverz matrice A:")
