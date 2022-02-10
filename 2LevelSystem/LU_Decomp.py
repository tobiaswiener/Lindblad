import numpy as np
import scipy as sp
from scipy import linalg

np.set_printoptions(suppress=True)
# set terminal width
np.set_printoptions(linewidth=800)
n = 4
for i in range(100):
    # L = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    A = np.diag([7 +1j, 1, 4 + 2j, 3, -1, 8 , 4 + 2j, 4 + 2j])
    U = np.random.rand(A.shape[0], A.shape[1])
    U_inv = sp.linalg.inv(U)
    B = np.einsum("ij,jk,kl->il", U_inv, A, U)

    values, Q, P = sp.linalg.eig(B, left=True)

    M = np.einsum("ij,jk->ik", Q.T.conj(), P)
    if not np.allclose(M,np.diag(np.diag(M))):
        print(i)
        print(M)
        break

l, u = sp.linalg.lu(M, permute_l=True)
l_inv = sp.linalg.inv(l)
u_inv = sp.linalg.inv(u)

p_new = np.einsum("ij,jk->ik",P,u_inv)
q_new = np.einsum("ij,jk->ik",l_inv,Q)

#diag = np.einsum("ij,jk,kl,lm->im", l_inv, Q.T.conj(), P, u_inv)

print(np.allclose(np.einsum("ij,jk->ik", q_new.T.conj(), B), np.einsum("ij,jk->ik", np.diag(values), q_new.T.conj())))
print(np.allclose(np.einsum("ij,jk->ik", B, p_new), np.einsum("ij,jk->ik", p_new, np.diag(values))))
