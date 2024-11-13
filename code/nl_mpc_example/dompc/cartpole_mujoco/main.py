import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

# import model and controller
from model import model_set
from controller import control

def run_onestep(x0, h, delta_t):
    
    #set up system and controller
    model = model_set(x0, h)
    mpc = control(model, delta_t)

    estimator = do_mpc.estimator.StateFeedback(model)
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.1)
    simulator.setup()

    # Initial state
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    # preform steps
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    curx = x0[0]
    curtheta = x0[1]
    curdx = x0[2]
    curdtheta = x0[3]
    curf = u0[0]
    J = (curtheta - np.pi)**2

    state = np.matrix([curx, curtheta, curdx, curdtheta])

    return state, curf, J





if __name__ == '__main__':
    # set simulation parameters
    num_steps = 160
    delta_t = .1
    h = delta_t

    # initial condition
    x0 = 0
    theta0 = np.deg2rad(0)
    dx0 = 0
    dtheta0 = np.deg2rad(0)
    X0 = np.matrix([[x0], [theta0], [dx0], [dtheta0]])

    # set matrices
    xarr = []
    thetaarr = []
    farr = []
    jarr = []
    tarr = []

    for k in range(num_steps):
        state, f, J = run_onestep(X0, h, delta_t)
        curx = state[0]
        curtheta = state[1]
        cur_t = delta_t*k
        print(k)
        
        #if k % 5 == 0:
        xarr = np.append(xarr, curx)
        thetaarr = np.append(thetaarr, curtheta)
        farr = np.append(farr, f)
        jarr = np.append(jarr, J)
        tarr = np.append(tarr, cur_t)
        

        X0 = state


    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('States and Controls Over Entire Range')
    fig.tight_layout()

    # position states
    ax1.plot(tarr, xarr, label='X')
    ax2.plot(tarr, thetaarr)
    ax3.plot(tarr, farr)
    ax4.plot(tarr, jarr)
    
    ax1.set_ylabel('X')
    ax2.set_ylabel('Theta')
    ax3.set_ylabel('F')
    ax4.set_ylabel('Cost')

    ax3.set_xlabel('Time')
    plt.savefig('cartpole_ts', bbox_inches='tight')



    """from matplotlib import rcParams
    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18

    import matplotlib.pyplot as plt
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16,9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.show()"""





