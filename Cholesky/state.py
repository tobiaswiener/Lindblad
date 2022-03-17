from basis import PauliBasis
import numpy as np
import math
from collections import defaultdict

SEED = 1234

np.set_printoptions(precision=2, suppress=True)


class State:
    rng = np.random.default_rng(SEED)

    def __init__(self, n=1, c=None, ):
        if c == None:
            c = {}
        self.basis = PauliBasis(n)

        assert State._c_is_valid(n, c), "coefficient dict is not valid."

        self.c = c

    @staticmethod
    def inner_product(a, b):
        assert a.basis.basis_states.keys() == b.basis.basis_states.keys(), "a and b need to have the same basis"

        inner = 0
        for j in a.basis.basis_states.keys():
            if (j in a.c) and (j in b.c):
                inner += a.c[j] * b.c[j]
        return inner

    @staticmethod
    def inner_product_matrix(ma, mb):
        assert ma.shape == mb.shape, "ma and mb need to have the same shape"
        inner = 1 / (2 ** n) * np.einsum("ij,ji", ma.T.conj(), mb)
        return inner

    @classmethod
    def _c_is_valid(cls, n, c):
        basis = PauliBasis(n)
        valid = True
        for j in c.keys():
            if j not in basis.basis_states:
                valid = False
        return valid

    @classmethod
    def from_matrix(cls, matrix):
        assert matrix.shape[0] == matrix.shape[1], "matrix must be square"
        assert math.log(matrix.shape[0], 2).is_integer(), "matrix dimension must be power of 2!"
        n = int(math.log(matrix.shape[0], 2))
        basis = PauliBasis(n)
        c = basis.matrix_to_coefficients(matrix)
        return cls(n, c)

    @classmethod
    def from_random(cls, n):
        basis = PauliBasis(n, )
        c = {}
        for j in basis.basis_states.keys():
            c[j] = cls.rng.random()
        return cls(n, c)

    def get_matrix(self):
        return self.basis.coefficients_to_matrix(self.c)

    def is_pos_definite(self):
        return np.all(np.linalg.eigvals(self.get_matrix()) > 0)


if __name__ == '__main__':
    n = 1
    omega = 0.5
    energy = 1.
    h = np.array([[0, omega],
                  [omega, energy]])
    H = 1.
    for i in range(n):
        H = np.kron(H,h)



