import itertools
import numpy as np

np.set_printoptions(precision=2, suppress=True)


class PauliHilbertBasis:
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

        assert PauliHilbertBasis._is_orthonormal(basis=basis), "Basis is not orthonormal!"
        self.basis = basis

    @staticmethod
    def _is_orthonormal(basis):
        for (mu, nu) in itertools.product(basis, basis):
            norm = np.trace(basis[mu] @ basis[nu])
            if mu == nu:
                if not np.isclose(norm, 1):
                    return False
            else:
                if not np.isclose(norm, 0):
                    return False

        return True

    def get_basis_element(self, mu):
        basis_element = 1.
        for i in range(self.n):
            basis_element = np.kron(basis_element, self.s_norm[mu[i]])
        norm = np.trace(basis_element @ basis_element)
        assert np.isclose(norm, 1)
        return basis_element

    def _get_coefficient(self, M, mu):
        c_mu = np.einsum("ij,ji", M, self.basis[mu])
        return c_mu

    def matrix_to_coefficient(self, M):
        hermitian = np.allclose(M, M.conj().T)
        c = {}
        for mu in self.basis.keys():
            c_mu = self._get_coefficient(M, mu)
            if hermitian and np.isclose(c_mu.imag, 0):
                c_mu = c_mu.real
            c[mu] = c_mu
        return c

    def coefficient_to_matrix(self, c):
        M = np.zeros((self.dim_hilbert, self.dim_hilbert), dtype=complex)
        for mu, c_mu in c.items():
            M += c_mu * self.basis[mu]
        return M


def make_random_hermitian_matrix(n):
    dim = 2 ** n
    H = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    H = H + H.conjugate().T
    return H


if __name__ == '__main__':
    b1 = PauliHilbertBasis(1)
    H1 = make_random_hermitian_matrix(1)

    b2 = PauliHilbertBasis(2)
    H2 = make_random_hermitian_matrix(2)

    b3 = PauliHilbertBasis(3)
    H3 = make_random_hermitian_matrix(3)
