import matplotlib.pyplot as plt
import numpy as np



def plot_rho_to(t, rho_t,steady_state):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Two-level system with decay")

    ax1.plot(t, rho_t[0].real, label=r"$\rho_{00}(t)$",color="tab:blue")
    ax1.plot(t, rho_t[3].real, label=r"$\rho_{11}(t)$",color="tab:red")
    ax1.axhline(y=steady_state[0].real,label=r"$\rho^{\mathrm{ss}}_{00}$",color="tab:blue", linestyle='--')
    ax1.axhline(y=steady_state[3].real,label=r"$\rho^{\mathrm{ss}}_{11}$",color="tab:red", linestyle='--')

    ax1.set(xlabel="time", ylabel="population")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.plot(t, np.abs(rho_t[1]), label=r"$abs(\rho_{01})$",color="tab:blue")
    ax2.plot(t, np.angle(rho_t[1]), label=r"$arg(\rho_{01})$",color="tab:red")
    ax2.axhline(y=np.abs(steady_state[1]),label=r"$abs(\rho^{\mathrm{ss}}_{01})$",color="tab:blue", linestyle='--')
    ax2.set(xlabel="time", ylabel="coherence")
    ax2.legend()
    plt.show()