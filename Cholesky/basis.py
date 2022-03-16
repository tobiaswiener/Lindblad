import itertools

import numpy as np



class PauliBasis():
    pauli_idxs = (0, 1, 2, 3)
    s0 = np.matrix([[1, 0], [0, 1]])
    s1 = np.matrix([[0, 1], [1, 0]])
    s2 = np.matrix([[0, -1j], [1j, 0]])
    s3 = np.matrix([[1, 0], [0, -1]])
    s = np.array((s0, s1, s2, s3))

    def __init__(self, n):



        self.n = n
        self.dim_hilbert = 2**n
        self.dim_liouville = 4**n

        self.basis_states = {}
        self._init_basis_states()



    def _init_basis_states(self):
        for j in itertools.product(self.pauli_idxs, repeat=self.n):
            self.basis_states[j] = self._make_basis_element(j)

    def _make_basis_element(self, j):
        sigma = 1.
        for k in range(self.n):
            sigma = np.kron(sigma, self.s[j[k]])
        return sigma

    def _get_coefficient(self, j, M):
        cj = 1/self.dim_hilbert*np.einsum("ij,ji", self.basis_states[j], M)
        return cj

    def matrix_to_coefficients(self, M):
        c = {}
        for j, sigma in self.basis_states.items():
            cj = self._get_coefficient(j, M)
            c[j] = cj

        return c

    def coefficients_to_matrix(self, c):
        M = np.zeros((self.dim_hilbert, self.dim_hilbert), dtype=complex)
        for j, c_j in c.items():
            M += c_j * self.basis_states[j]

        M = M
        return M



if __name__ == '__main__':
    basis = PauliBasis(6)
