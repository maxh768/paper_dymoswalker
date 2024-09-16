import numpy as np
import matplotlib.pyplot as plt

# timeseries plots for all states
def plot_timeseries(x1arr, x2arr, x3arr, x4arr, x5arr, x6arr, timearr, dir='./research_template/threeleg_graphs/', name='timeseries_data'):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('States and Controls Over Entire Range')
    fig.tight_layout()

    # position states
    ax1.plot(timearr, x1arr, label='Theta Stance')
    ax1.plot(timearr, x2arr, label='Theta Hip')
    ax1.plot(timearr, x3arr, label='Theta Thigh')
    ax1.legend()

    # velocity states
    ax2.plot(timearr, x4arr, label='alpha stance')
    ax2.plot(timearr, x5arr, label='alpha hip')
    ax2.plot(timearr, x6arr, label='alpha thigh')
    ax2.legend()

    #control
    #ax3.plot(timearr, tauarr)
  
    ax1.set_ylabel('Angle (rad)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    #ax3.set_ylabel('Tau (N*m)')
    ax2.set_xlabel('time (s)')
    

    plt.savefig(dir+name, bbox_inches='tight')
