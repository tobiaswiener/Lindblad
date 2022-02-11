import numpy as np
import scipy as sp
from scipy import linalg

import plotting
import utils

# suppress small values when printing
np.set_printoptions(suppress=True)
# set terminal width
np.set_printoptions(linewidth=800)

DIM_HILBERT_SPACE = 2
DIM_LIOUVILLE_SPACE = DIM_HILBERT_SPACE ** 2


def biorthonormalize_Q_P(Q, P):
    """
    source: https://joshuagoings.com/2015/04/03/biorthogonalizing-left-and-right-eigenvectors-the-easy-lazy-way/

    :param Q:
    :param P:
    :return:
    """
    M = np.einsum("ij,jk->ik", Q.T.conj(), P)
    l, u = sp.linalg.lu(M, permute_l=True)
    l_inv = sp.linalg.inv(l)
    u_inv = sp.linalg.inv(u)

    P_new = np.einsum("ij,jk->ik", P, u_inv)
    Q_new = np.einsum("ij,jk->ik", l_inv, Q)
    return P_new, Q_new


def make_liouvillian(energy, omega, gamma):
    """
    Source: https://aip.scitation.org/doi/10.1063/1.5115323
    Builds the Liouviallian operator corresponding to the Lindblad equation for the two-level System with decay.
    :param energy: Energy of the excited State.
    :param omega: Coherently driving frequency between the two states.
    :param gamma: Coupling between the two states and the vacuum.
    :return:
        - L - Liouvillian matrix
    """
    L = np.array([[0, 1j * omega, -1j * omega, gamma],
                  [1j * omega, 1j * energy - gamma / 2, 0, -1j * omega],
                  [-1j * omega, 0, -1j * energy - gamma / 2, 1j * omega],
                  [0, -1j * omega, 1j * omega, -gamma]])
    return L


def get_steady_state(eigvals, eigvecs):
    ss_index = np.argwhere(np.isclose(np.abs(eigvals), 0))[0][0]
    steady_state = eigvecs[:, ss_index] / np.abs(eigvecs[:, ss_index][0] + eigvecs[:, ss_index][3])
    return steady_state


def expand_rho_t(eigvals, Q, P, rho_0, t):
    exp_lambda_t = np.exp(np.einsum("i,t->it", eigvals, t))
    rho_t = np.einsum("it,mi,iv,v->mt", exp_lambda_t, P, Q.conj().T, rho_0)

    return rho_t


def reshape_rho_t_to_matrix(rho_t):
    rho_t_matrix = np.reshape(rho_t, (DIM_HILBERT_SPACE, DIM_HILBERT_SPACE, -1))
    return rho_t_matrix


def check_density_matrix(rho_t):
    rho_t_matrix = reshape_rho_t_to_matrix(rho_t=rho_t)
    trace_rho_t = np.einsum("iit->t", rho_t_matrix)
    if not np.allclose(trace_rho_t, 1):
        raise utils.UnphysicalDensityMatrixException("Trace of density matrix is not one")
    if not np.allclose(rho_t_matrix, np.transpose(rho_t_matrix, axes=(1, 0, 2)).conj()):
        raise utils.UnphysicalDensityMatrixException("Density matrix is not hermitian")


if __name__ == '__main__':
    energy = 1.
    omega = 0.9
    gamma = 0.1

    rho_00 = 0.
    rho_01 = 0.
    rho_10 = 0.
    rho_11 = 1.
    rho_0 = np.array([rho_00, rho_01, rho_10, rho_11])
    t = np.linspace(0, 100, 1000)

    L = make_liouvillian(energy, omega, gamma)

    eigvals, Q, P = sp.linalg.eig(L, left=True, right=True)
    P_biorthonorm, Q_biorthonorm = biorthonormalize_Q_P(Q, P)
    rho_t = expand_rho_t(eigvals=eigvals, Q=Q_biorthonorm, P=P_biorthonorm, rho_0=rho_0, t=t)

    check_density_matrix(rho_t)

    steady_state = get_steady_state(eigvals=eigvals, eigvecs=P)
    plotting.plot_rho_to(t=t, rho_t=rho_t, steady_state=steady_state)
