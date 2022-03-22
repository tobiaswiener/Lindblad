import numpy as np
np.set_printoptions(precision=2, suppress=True)

from basis import Basis
from paulihilbertbasis import PauliHilbertBasis
from pauliliouvillebasis import PauliLiouvilleBasis
def make_random_hermitian_matrix(n):
    dim = 2 ** n
    H = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    H = H + H.conjugate().T
    return H

def make_random_matrix(n):
    dim = 2 ** n
    H = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    return H
if __name__ == '__main__':
    n = 3
    hb = PauliHilbertBasis(n)
    lb = PauliLiouvilleBasis(n)

    H1 = make_random_matrix(n)

    H1_sk = lb._bra_flip(H1)


    H1_new = hb.coefficient_to_operator(hb.operator_to_coefficient(H1))
    H1_sk_new = lb.coefficients_to_superket(lb.superket_to_coefficients(H1_sk))

