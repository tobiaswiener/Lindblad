import numpy as np
import matplotlib.pyplot as plt

# suppress small values wenn printing
np.set_printoptions(suppress=True)
# set terminal width
np.set_printoptions(linewidth=800)

DIM_HILBERT_SPACE = 2
DIM_LIOUVILLE_SPACE = DIM_HILBERT_SPACE ** 2

basis = np.eye(DIM_LIOUVILLE_SPACE)

OMEGA = 1.
ENERGY = 1.

t = np.linspace(0, 5, 100)


def make_liouvillian(energy, omega):
    L = -1j * np.array([[0, -omega, omega, 0],
                        [-omega, -energy, 0, omega],
                        [omega, 0, energy, -omega],
                        [0, omega, -omega, 0]])
    return L


def transform_matrix(M, V, V_inv):
    M_new = np.matmul(V_inv, np.matmul(M, V))
    return M_new

def transform_vektor(vec, V_inv):
    vec_new = np.matmul(V_inv,vec)
    return vec_new




if __name__ == '__main__':
    energy = 1
    omega = 1
    rho_00 = 0.
    rho_01 = 0.
    rho_10 = 0.
    rho_11 = 1.


    L = make_liouvillian(energy,omega)
    eigval, V = np.linalg.eig(L)
    V_inv = np.linalg.inv(V)
    L_tilde = transform_matrix(L, V, V_inv)
    rho_0 = np.array([rho_00, rho_01, rho_10, rho_11])
    rho_0_t = np.zeros((rho_0.size, t.size), dtype=np.complex64) + rho_0[:, np.newaxis]
    rho_t = np.zeros_like(rho_0_t)


    U_tilde = np.zeros((DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE, t.size), dtype=np.complex64)
    U = np.zeros((DIM_LIOUVILLE_SPACE, DIM_LIOUVILLE_SPACE, t.size), dtype=np.complex64)
    for i in range(DIM_LIOUVILLE_SPACE):
        U_tilde[i, i, :] = np.exp(eigval[i] * t)

    for i in range(len(t)):
        U[:,:,i] = V@U_tilde[:,:,i]@V_inv

    for i in range(len(t)):
        rho_t[:,i] = U[:,:,i]@rho_0_t[:,i]

    plt.plot(t, rho_t[0])
    plt.plot(t, rho_t[3])
    plt.show()