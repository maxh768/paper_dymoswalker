import numpy as np
import matplotlib.pyplot as plt

def pl_ts(th1, th2, dth1, dth2, mpc1, mpc2, dmpc1, dmpc2, u1, u2, tarr ,name='doubpem_mjpc_times'):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('States and Controls Over Entire Range')
    fig.tight_layout()

    # position states
    ax1.plot(tarr, th1,color='b', label='theta1: MuJoCo')
    ax1.plot(tarr, mpc1, '--',label='theta1: do-mpc',color='k')
    ax1.plot(tarr, th2, label='theta2: MuJoCo',color='g')
    ax1.plot(tarr, mpc2, '--',label='theta2: do-mpc', color='olive')

    ax2.plot(tarr, dth1, label='dtheta1: MuJoCo',color='b')
    ax2.plot(tarr, dmpc1, '--',label='dtheta1: do-mpc',color='k')
    ax2.plot(tarr, dth2, label='dtheta2: MuJoCo', color='g')
    ax2.plot(tarr, dmpc2, '--',label='dtheta2: do-mpc', color='olive')

    ax3.plot(tarr, u1, label='Lower Input')
    ax3.plot(tarr, u2, label='Upper Input')
    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_ylabel('Angle')
    ax2.set_ylabel('Angular Velocity')
    ax3.set_ylabel('Control Input')

    ax3.set_xlabel('Time')
    plt.savefig(name, bbox_inches='tight')