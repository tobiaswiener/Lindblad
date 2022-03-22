import itertools
from abc import ABCMeta, abstractmethod
import numpy as np
class Basis(metaclass=ABCMeta):
    local_idxs = (0, 1, 2, 3)
    s0 = np.matrix([[1, 0], [0, 1]])
    s1 = np.matrix([[0, 1], [1, 0]])
    s2 = np.matrix([[0, -1j], [1j, 0]])
    s3 = np.matrix([[1, 0], [0, -1]])
    s = np.array((s0, s1, s2, s3))
    s_norm = 1 / np.sqrt(2) * np.array((s0, s1, s2, s3))
    def __init__(self, n):
        self.n = n
        self.dim_hilbert = 2 ** n
        self.dim_liouville = self.dim_hilbert**2

        self.indices = []
        self._init_indices()

        self.basis_elements = {}
        self._init_basis_elements()

    def _init_indices(self):
        for mu in itertools.product(self.local_idxs, repeat=self.n):
            self.indices.append(mu)

    def _init_basis_elements(self):
        basis_elements = {}
        for mu in self.indices:
            basis_elements[mu] = self._make_basis_element(mu)
        self.basis_elements = basis_elements

    @abstractmethod
    def inner(self, A,B):
        pass

    @abstractmethod
    def _make_basis_element(self, mu):
        pass

    def _bra_flip(self, M):
        assert M.shape == (self.dim_hilbert, self.dim_hilbert), "matrix must have shape (n**2,n**2)"
        M_sk = M.flatten()
        return M_sk

    def _bra_flip_inv(self, M_sk):
        assert M_sk.shape == (self.dim_liouville,), "superket must have shape (n**4)"
        return M_sk.reshape(self.dim_hilbert, self.dim_hilbert)