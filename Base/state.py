from paulihilbertbasis import PauliHilbertBasis
from pauliliouvillebasis import PauliLiouvilleBasis
import basis

import itertools
import numpy as np
from basis import Basis
import math

np.set_printoptions(precision=2, suppress=True)



class State:
    rng = np.random.default_rng(
    )

    def __init__(self, n=1, rho_mu=None, ):
        self.n = n
        if rho_mu == None:
            rho_mu = {}
        self.basis = PauliHilbertBasis(n)

        assert State._rho_mu_is_valid(n, rho_mu), "coefficient dict is not valid."

        self.rho_mu = rho_mu
        self.rho_matrix = self.basis.coefficient_to_operator(rho_mu)

        self.v = None
        self.v_mu = None



    @classmethod
    def _rho_mu_is_valid(cls, n, c):
        basis = PauliHilbertBasis(n)
        valid = True
        for j in c.keys():
            if j not in basis.basis_elements:
                valid = False
        return valid


    @classmethod
    def from_rho(cls, rho_matrix):
        assert rho_matrix.shape[0] == rho_matrix.shape[1], "matrix must be square"
        assert math.log(rho_matrix.shape[0], 2).is_integer(), "matrix dimension must be power of 2!"
        n = int(math.log(rho_matrix.shape[0], 2))
        basis = PauliHilbertBasis(n)
        rho_mu = basis.operator_to_coefficient(rho_matrix)
        state = cls(n, rho_mu)
        return state

    @classmethod
    def from_random(cls, n):
        basis = PauliHilbertBasis(n)

        dim = 2 ** n
        v = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
        v = v + v.T.conj()
        v_mu = basis.operator_to_coefficient(v)

        tr = 0.
        for key,value in v_mu.items():
            tr += 2*value**2

        for key,value in v_mu.items():
            v_mu[key] = value/np.sqrt(2*tr.real)

        v = basis.coefficient_to_operator(v_mu)


        rho = v @ (v.T.conj())
        rho = rho
        state = State.from_rho(rho_matrix=rho)
        state.v = v
        state.v_mu = v_mu
        return state




    def get_matrix(self):
        return self.basis.coefficient_to_operator(self.rho_mu)

    def is_pos_definite(self):
        return np.all(np.linalg.eigvals(self.get_matrix()) > 0)




