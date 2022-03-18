import itertools
import numpy as np
from paulihilbertbasis import PauliHilbertBasis
from basis import Basis

np.set_printoptions(precision=2, suppress=True)


class PauliLiouvilleBasis(Basis):
    def __init__(self, n):
        super().__init__(n)

    def _make_basis_element(self, mu):
        basis_element = 1.
        for i in range(self.n):
            basis_element = np.kron(basis_element, self.s_norm[mu[i]])
        return basis_element.flatten()

    def inner(self, A, B):
        inner = np.einsum("i,i", B, A.conj().T)
        return inner

    def superket_to_coefficients(self, M_sk):
        c = {}
        for mu in self.basis_elements.keys():
            c_mu = self.inner(self.basis_elements[mu], M_sk)
            c[mu] = c_mu
        return c

    def coefficients_to_superket(self, c):
        M_sk = np.zeros((self.dim_liouville,), dtype=complex)
        for mu, c_mu in c.items():
            M_sk += c_mu * self.basis_elements[mu]
        return M_sk


