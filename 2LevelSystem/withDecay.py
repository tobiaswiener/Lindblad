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


if __name__ == '__main__':
    energy = 1
    omega = 0.5
    gamma = 0.2
    rho_00 = 0.
    rho_01 = 0.
    rho_10 = 0.
    rho_11 = 1.
    rho_0 = np.array([rho_00, rho_01, rho_10, rho_11])

    t = np.linspace(0, 5, 100)

    L = make_liouvillian(energy, omega, gamma)
    eigval, V = np.linalg.eig(L)
    V_inv = np.linalg.inv(V)
    L_tilde = transform_matrix(L, V, V_inv)
    rho_0 = np.array([rho_00, rho_01, rho_10, rho_11])

    U_tilde = np.zeros((DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE, t.size), dtype=np.complex64)

    for i in range(DIM_LIOUVILLE_SPACE):
        U_tilde[i, i, :] = np.exp(eigval[i] * t)


    U = np.einsum("ik,klt,lj->ijt", V, U_tilde, V_inv)
    rho_t = np.einsum("ijt,j->it", U, rho_0)

    ## check: diagonal elements of density matrix sum up to one
    assert np.allclose(rho_t[0, :] + rho_t[3, :], np.ones_like(rho_t[0, :]))
    assert np.allclose(rho_t[1, :] - rho_t[2, :].T.conj(), np.zeros_like(rho_t[1, :]))

    # plot populations and coherences
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Two-level system with decay")

    ax1.plot(t, rho_t[0].real, label=r"$\rho_{00}$")
    ax1.plot(t, rho_t[3].real, label=r"$\rho_{11}$")
    ax1.set(xlabel="time", ylabel="population")
    ax1.legend()
    ax2.plot(t, np.abs(rho_t[1]), label=r"$abs(\rho_{01})$")
    ax2.plot(t, np.angle(rho_t[1]),  label=r"$arg(\rho_{01})$")
    ax2.set(xlabel="time", ylabel="coherence")
    ax2.legend()
    plt.show()