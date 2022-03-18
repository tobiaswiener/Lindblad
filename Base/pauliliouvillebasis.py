import itertools
import numpy as np
from paulihilbertbasis import PauliHilbertBasis

np.set_printoptions(precision=2, suppress=True)


class PauliLiouvilleBasis:
    pauli_local_idxs = (0, 1, 2, 3)
    s0 = np.matrix([[1, 0], [0, 1]])
    s1 = np.matrix([[0, 1], [1, 0]])
    s2 = np.matrix([[0, -1j], [1j, 0]])
    s3 = np.matrix([[1, 0], [0, -1]])
    s = np.array((s0, s1, s2, s3))
    s_norm = 1 / np.sqrt(2) * np.array((s0, s1, s2, s3))

    def __init__(self, n):
        self.n = n
        self.dim_hilbert = 2 ** n
        self.dim_liouville = 4 ** n

        self.indices = []
        self._init_indices()

        self.basis = {}
        self._init_basis()

    def _init_indices(self):
        for mu in itertools.product(self.pauli_local_idxs, repeat=self.n):
            self.indices.append(mu)

    def _init_basis(self):
        basis = {}

        for mu in self.indices:
            basis[mu] = self.get_basis_element(mu)
        assert PauliLiouvilleBasis._is_orthonormal(basis=basis), "Basis is not orthonormal!"
        self.basis = basis

    def get_basis_element(self, mu):
        basis_element = 1.
        for i in range(self.n):
            basis_element = np.kron(basis_element, self.s_norm[mu[i]])

        return basis_element.flatten()

    @staticmethod
    def _is_orthonormal(basis):
        for (mu, nu) in itertools.product(basis, basis):
            norm = np.dot(basis[mu].conj(), basis[nu])
            if mu == nu:
                if not np.isclose(norm, 1):
                    return False
            else:
                if not np.isclose(norm, 0):
                    return False
        return True

    def _get_coefficient(self, M_sk, mu):
        c_mu = np.einsum("i,i",self.basis[mu].conj(), M_sk)
        return c_mu

    def matrix_to_pauli_coefficient(self, M_sk):
        M = self._bra_flip_inv(M_sk)
        hermitian = np.allclose(M, M.conj().T)
        c = {}
        for mu in self.basis.keys():
            c_mu = self._get_coefficient(M_sk, mu)
            if hermitian and np.isclose(c_mu.imag, 0):
                c_mu = c_mu.real
            c[mu] = c_mu
        return c

    def pauli_coefficient_to_matrix(self, c):
        M_sk = np.zeros((self.dim_liouville,), dtype=complex)
        for mu, c_mu in c.items():
            M_sk += c_mu * self.basis[mu]
        return M_sk

    def _bra_flip(self, M):
        assert M.shape == (self.dim_hilbert, self.dim_hilbert), "matrix must have shape (n**2,n**2)"
        M_sk = M.flatten()
        return M_sk
    def _bra_flip_inv(self, M_sk):
        assert M_sk.shape == (self.dim_liouville,), "superket must have shape (n**4)"
        return M_sk.reshape(self.dim_hilbert, self.dim_hilbert)


def make_random_hermitian_matrix(n):
    dim = 2 ** n
    H = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    H = H + H.conjugate().T
    return H


if __name__ == '__main__':
    b1 = PauliLiouvilleBasis(1)
    H1 = make_random_hermitian_matrix(1)

    b2 = PauliLiouvilleBasis(2)
    H2 = make_random_hermitian_matrix(2)

    b3 = PauliLiouvilleBasis(3)
    hb3 = PauliHilbertBasis(3)

    H3 = make_random_hermitian_matrix(3)


