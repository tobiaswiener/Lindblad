import itertools

import numpy as np
from basis import PauliBasis


class SuperPauliBasis:

    def __init__(self, n):
        self.n = n
        self.dim_hilbert = 2**n
        self.dim_liouville = 4**n
        self.pauli_basis = PauliBasis(n)
        self.operator_basis = {}
        self._init_operator_basis()

    def _init_operator_basis(self):
        for alpha, beta in itertools.product(self.pauli_basis.basis_states, self.pauli_basis.basis_states):
            self.operator_basis[alpha, beta] = np.outer(self.pauli_basis.basis_states[alpha],
                                                   self.pauli_basis.basis_states[beta])


    def _get_coefficient(self, alpha, beta, operator):
        c_ab = 1/self.dim_liouville*np.einsum("i,ij,j",alpha, operator,beta)
        return c_ab

    def operator_to_coefficients(self, operator):
        hermitian = np.allclose(operator, operator.conj().T)
        c = {}

        for alpha,beta in self.operator_basis.keys():
            c_ab = self._get_coefficient(alpha,beta,operator)
            c[(alpha,beta)] = c_ab

        return c


    def coefficients_to_operator(self, c):
        op = np.zeros((self.dim_hilbert, self.dim_hilbert), dtype=complex)
        for (a,b), c_ab in c.items():
            op += c_ab*self.operator_basis[(a,b)]

        return op
