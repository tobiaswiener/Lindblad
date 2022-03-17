from basis import PauliBasis
from superbasis import SuperPauliBasis
import numpy as np
import math
from collections import defaultdict

SEED = 1234

np.set_printoptions(precision=2, suppress=True)




class Operator:

    def __init__(self,n=1, c=None):
        if c == None:
            c = {}
        self.basis = SuperPauliBasis(n)

        self.c = c

    @classmethod
    def from_operator(cls, operator):
        assert operator.shape[0] == operator.shape[1], "matrix must be square"
        assert math.log(operator.shape[0], 2).is_integer(), "matrix dimension must be power of 2!"
        n = int(math.log(operator.shape[0], 2))
        basis = SuperPauliBasis(n)
        c = basis.operator_to_coefficients(operator)
        return cls(n, c)




if __name__ == '__main__':
    omega = 0.5
    energy = 1.
    h = np.array([[0,omega],
                  [omega,energy]])

    n = 1
    H = 1.
    L = np.kron(h,np.ones_like(h))-np.kron(np.ones_like(h),h)

    op = Operator.from_operator(L)