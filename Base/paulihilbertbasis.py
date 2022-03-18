import itertools
import numpy as np
from basis import Basis

np.set_printoptions(precision=2, suppress=True)


class PauliHilbertBasis(Basis):
    def __init__(self, n):
        super().__init__(n)
        self.n = n
        self.dim_hilbert = 2 ** n

        self._init_basis_elements()

    def _make_basis_element(self, mu):
        basis_element = 1.
        for i in range(self.n):
            basis_element = np.kron(basis_element, self.s_norm[mu[i]])
        return basis_element

    def inner(self, A,B):
        inner = np.einsum("ij,ji",B,A.conj().T)
        return inner

    def operator_to_coefficient(self, M):
        c = {}
        for mu in self.basis_elements.keys():
            c_mu = self.inner(self.basis_elements[mu], M)
            c[mu] = c_mu
        return c

    def coefficient_to_operator(self, c):
        M = np.zeros((self.dim_hilbert, self.dim_hilbert), dtype=complex)
        for mu, c_mu in c.items():
            M += c_mu * self.basis_elements[mu]
        return M


