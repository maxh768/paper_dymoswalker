import numpy as np
import matplotlib.pyplot as plt

def pl_ts(xarr, tarr, thetaarr, the_dmpc, yarr, farr,name='cartpole_mjpc_times'):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('States and Controls Over Entire Range')
    fig.tight_layout()

    # position states
    ax1.plot(tarr, xarr, label='MuJoCo')
    ax1.plot(tarr, yarr, '--',label='do-mpc')
    ax2.plot(tarr, thetaarr, label='MuJoCo')
    ax2.plot(tarr, the_dmpc, '--',label='do-mpc')
    ax3.plot(tarr, farr)
    ax1.legend()
    ax2.legend()

    ax1.set_ylabel('X')
    ax2.set_ylabel('Theta')
    ax3.set_ylabel('F')

    ax3.set_xlabel('Time')
    plt.savefig(name, bbox_inches='tight')