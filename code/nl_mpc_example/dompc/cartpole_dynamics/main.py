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

m = 5
M = 10.5
L = 0.3

num_steps = 400

delta_t = .04
model = model_set(M,m,L)
mpc = control(model, delta_t)

# estimator and simulator (need to replace with mujoco)
estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = delta_t)
simulator.setup()

x0 = np.array([0, 0, 0, 0])
# Initial state
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()
u0 = 0

# control   
xarr = []
thetaarr = []
farr = []
tarr = []
for i in range(num_steps):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print(i)

    curx = float(x0[0])
    curtheta = float(x0[2])
    curf = float(u0)
    curt = i*delta_t

    xarr.append(curx)
    thetaarr.append(curtheta)
    farr.append(curf)
    tarr.append(curt)

from animate_cartpole import animate_cartpole
animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=L, save_gif=True, name='cartpole_mjpc.gif')

import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('States and Controls Over Entire Range')
fig.tight_layout()

# position states
ax1.plot(tarr, xarr)
ax2.plot(tarr, thetaarr)
ax3.plot(tarr, farr)

ax1.set_ylabel('X')
ax2.set_ylabel('Theta')
ax3.set_ylabel('F')

ax3.set_xlabel('Time')
plt.savefig('timeseries_dynamics', bbox_inches='tight')






