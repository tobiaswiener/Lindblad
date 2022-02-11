import numpy as np
import scipy as sp
from scipy import linalg

import plotting
import utils

# suppress small values when printing
np.set_printoptions(suppress=True)
# set terminal width
np.set_printoptions(linewidth=800)


def biorthonormalize_P_Q(P, Q):
    """
    Biorthonormalizes the left and right eigenvectors i.e. :math:`P^\dagger Q = \mathbb{1}`

    Parameters
    ----------
    P: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Matrix of normalized left eigenvectors, where P[:,i] is the ith left eigenKET.
    Q: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Matrix of normalized right eigenvectors where Q[:,i] is the ith right eigenKET.

    Returns
    -------
    P: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Matrix of biorthonormalized left eigenvectors, where P[:,i] is the ith left eigenKET.

    Notes
    -------
    source: https://joshuagoings.com/2015/04/03/biorthogonalizing-left-and-right-eigenvectors-the-easy-lazy-way/
    """

    M = np.einsum("ij,jk->ik", Q.T.conj(), P)
    l, u = sp.linalg.lu(M, permute_l=True)
    l_inv = sp.linalg.inv(l)
    u_inv = sp.linalg.inv(u)

    P_bi = np.einsum("ij,jk->ik", P, u_inv)
    Q_bi = np.einsum("ij,jk->ik", l_inv, Q)
    return P_bi, Q_bi


def make_liouvillian(energy, omega, gamma):
    """
     Builds the Liouviallian operator corresponding to the Lindblad equation for the two-level System with decay.
    Parameters
    ----------
    energy: float
        Energy of the excited State.
    omega: float
        Coherently driving frequency between the two states.
    gamma: float
        Coupling between the two states and the vacuum.

    Returns
    -------
    L: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Liouvillian matrix

    Notes
    -----
    Source: https://aip.scitation.org/doi/10.1063/1.5115323
    """

    L = np.array([[0, 1j * omega, -1j * omega, gamma],
                  [1j * omega, 1j * energy - gamma / 2, 0, -1j * omega],
                  [-1j * omega, 0, -1j * energy - gamma / 2, 1j * omega],
                  [0, -1j * omega, 1j * omega, -gamma]])
    return L




def get_steady_state(eigvals, eigvecs):
    """
    Extracts the steady state of the Liouvillian out of the eigenstates.
    The steady state is the state corresponding to the zero eigenvalue.
    Parameters
    ----------
    eigvals: (DIM_LIOUVILLE_SPACE,) np.ndarray
        Eigenvalues of Liouvillian.
    Q: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Matrix of normalized right eigenvectors where Q[:,i] is the ith right eigenKET.

    Returns
    -------
    rho_ss: (DIM_LIOUVILLE_SPACE,) np.ndarray
        Density Matrix describing the steady state of the system as a vector in Fock-Liouville space.
    """

    ss_value = 0
    ss_index = np.argwhere(np.isclose(np.abs(eigvals), ss_value))[0][0]
    steady_state = eigvecs[:, ss_index] / np.abs(eigvecs[:, ss_index][0] + eigvecs[:, ss_index][3])
    return steady_state



def expand_rho_t(eigvals, P, Q, rho_0, t):
    """Expansion of time dependent density matrix in the eigenbasis of the Liouvillian.

    Parameters
    ----------
    eigvals: (DIM_LIOUVILLE_SPACE,) np.ndarray
        Eigenvalues of Liouvillian.
    P: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Matrix of biorthonormalized left eigenvectors, where P[:,i] is the ith left eigenKET.
    Q: (DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE) np.ndarray
        Matrix of biorthonormalized right eigenvectors where Q[:,i] is the ith right eigenKET.
    rho_0: (DIM_LIOUVILLE_SPACE,) np.ndarray
        Density matrix at t=0 as a vector in Fock-Liouville space.
    t: (T,) np.ndarray
        Array of time values.

    Returns
    -------
    rho_t: (DIM_LIOUVILLE_SPACE,T) np.ndarray
        Time dependent density matrix as a vector in Fock-Liouville space.

    Notes
    -----
    .. math:: | \rho(0) \rangle\rangle = \sum_i |\Lambda_i^{\mathrm{R}}\rangle\rangle \langle\langle \Lambda_i^{\mathrm{L}}|\rho(0)\rangle\ranlge

    """

    exp_lambda_t = np.exp(np.einsum("i,t->it", eigvals, t))
    rho_t = np.einsum("it,mi,iv,v->mt", exp_lambda_t, Q, P.conj().T, rho_0)

    return rho_t


def reshape_rho_t_to_matrix(rho_t):
    """Reshapes density matrix written as a vector in Fock-Liouville space to a matrix in hilbert space.

    Parameters
    ----------
    rho_t: (DIM_LIOUVILLE_SPACE,T) np.ndarray
        Time dependent density matrix as a vector in Fock-Liouville space.
    Returns
    -------
    rho_t_matrix: (DIM_HILBERT_SPACE,DIM_HILBERT_SPACE,T)
        Time dependent density matrix written as a matrix in hilbert space.
    """
    rho_t_matrix = np.reshape(rho_t, (DIM_HILBERT_SPACE, DIM_HILBERT_SPACE, -1))
    return rho_t_matrix


def check_density_matrix(rho_t):
    """Checks if density matrix has physical properties.

    All density matrices have unit trace and are hermitian.
    Parameters
    ----------
    rho_t: (DIM_LIOUVILLE_SPACE,T) np.ndarray

    Raises
    ------
    UnphysicalDensityMatrixError
        if trace of density matrix is not 1 at all times.
        if density matrix is not hermitian at all times.
    """
    rho_t_matrix = reshape_rho_t_to_matrix(rho_t=rho_t)
    trace_rho_t = np.einsum("iit->t", rho_t_matrix)
    if not np.allclose(trace_rho_t, 1):
        raise utils.UnphysicalDensityMatrixException("Trace of density matrix is not one")
    if not np.allclose(rho_t_matrix, np.transpose(rho_t_matrix, axes=(1, 0, 2)).conj()):
        raise utils.UnphysicalDensityMatrixException("Density matrix is not hermitian")


if __name__ == '__main__':
    DIM_HILBERT_SPACE = 2
    DIM_LIOUVILLE_SPACE = DIM_HILBERT_SPACE ** 2
    T = 100


    energy = 1.
    omega = 0.9
    gamma = 0.1

    rho_00 = 0.
    rho_01 = 0.
    rho_10 = 0.
    rho_11 = 1.
    rho_0 = np.array([rho_00, rho_01, rho_10, rho_11])
    t = np.linspace(0, T, 1000)

    L = make_liouvillian(energy, omega, gamma)

    eigvals, P, Q = sp.linalg.eig(L, left=True, right=True)
    P_biorthonorm, Q_biorthonorm = biorthonormalize_P_Q(P=P, Q=Q)
    rho_t = expand_rho_t(eigvals=eigvals, P=P_biorthonorm, Q=Q_biorthonorm, rho_0=rho_0, t=t)

    check_density_matrix(rho_t)

    steady_state = get_steady_state(eigvals=eigvals, eigvecs=P)
    plotting.plot_rho_to(t=t, rho_t=rho_t, steady_state=steady_state)
