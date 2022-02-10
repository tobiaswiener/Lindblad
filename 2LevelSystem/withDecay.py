import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg
# suppress small values when printing
np.set_printoptions(suppress=True)
# set terminal width
np.set_printoptions(linewidth=800)

DIM_HILBERT_SPACE = 2
DIM_LIOUVILLE_SPACE = DIM_HILBERT_SPACE ** 2

def norm_Q_P(Q, P):
    M = np.einsum("ij,jk->ik", Q.T.conj(), P)
    l, u = sp.linalg.lu(M, permute_l=True)
    l_inv = sp.linalg.inv(l)
    u_inv = sp.linalg.inv(u)

    P_new = np.einsum("ij,jk->ik", P, u_inv)
    Q_new = np.einsum("ij,jk->ik", l_inv, Q)
    return P_new, Q_new

def make_liouvillian(energy, omega, gamma):
    L = np.array([[0, 1j * omega, -1j * omega, gamma],
                  [1j * omega, 1j * energy - gamma/2, 0, -1j * omega],
                  [-1j * omega, 0, -1j * energy - gamma / 2, 1j * omega],
                  [0, -1j * omega, 1j * omega, -gamma]])
    return L


def transform_matrix(M, V, V_inv):
    M_new = np.einsum("ij,jk,kl->il", V_inv, M, V)
    return M_new


def transform_vektor(vec, V_inv):
    vec_new = np.einsum("ij,j->i", V_inv, vec)
    return vec_new

def right_eigenvalue_equation(L, right, eigvals):
    lhs = np.einsum("jk,ki->ji",L,right)
    rhs = np.einsum("i,ji->ji",eigvals,right)
    return np.allclose(lhs,rhs)
def left_eigenvalue_equation(L,left,eigvals):
    lhs = np.einsum("jk,ki->ji",L.conj().T,left)
    rhs = np.einsum("i,ji->ji",eigvals.conj(),left)
    return np.allclose(lhs,rhs)

def expand_rho(left,right,vector):
    left_vector = np.einsum("ki,k->i",left.conj(),vector)
    expanded = np.einsum("ki,i->k",right,left_vector)
    return expanded

def expand_rho_t(eigvals,left,right,rho_0,t):
    exp_eigvals = np.exp(np.einsum("i,t->it", eigvals, t))

    left_vector = np.einsum("ki,k->i",left.conj(),rho_0)
    rho_t = np.einsum("it,ki,i->kt",exp_eigvals,right,left_vector)

    return rho_t
if __name__ == '__main__':
    energy = 1
    omega = 1.
    gamma = 0.1
    rho_00 = 0.
    rho_01 = 0.
    rho_10 = 0.
    rho_11 = 1.

    rho_0 = np.array([rho_00, rho_01, rho_10, rho_11])

    t = np.linspace(0, 100, 1000)

    L = make_liouvillian(energy, omega, gamma)
    value, Q, P = sp.linalg.eig(L, left=True, right=True)
    P_biorthonorm ,Q_biorthonorm = norm_Q_P(Q,P)

    exp_lambda_t = np.exp(np.einsum("i,t->it", value, t))
    rho_t = np.einsum("it,mi,iv,v->mt", exp_lambda_t, P_biorthonorm, Q_biorthonorm.conj().T, rho_0)


    ## check: diagonal elements of density matrix sum up to one
    assert np.allclose(rho_t[0, :] + rho_t[3, :], np.ones_like(rho_t[0, :]))
    assert np.allclose(rho_t[1, :] - rho_t[2, :].T.conj(), np.zeros_like(rho_t[1, :]))

    # plot populations and coherences
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Two-level system with decay")

    ax1.plot(t, rho_t[0].real, label=r"$\rho_{00}$")
    ax1.plot(t, rho_t[3].real, label=r"$\rho_{11}$")
    ax1.set(xlabel="time", ylabel="population")
    ax1.set_ylim(0,1)
    ax1.legend()
    ax2.plot(t, np.abs(rho_t[1]), label=r"$abs(\rho_{01})$")
    ax2.plot(t, np.angle(rho_t[1]),  label=r"$arg(\rho_{01})$")
    ax2.set(xlabel="time", ylabel="coherence")
    ax2.legend()
    plt.show()
