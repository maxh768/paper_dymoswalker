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

m = 1
M = 5
L = 1

num_steps = 200

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
for i in range(num_steps):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print(i)

    curx = float(x0[0])
    curtheta = float(x0[1])
    curf = float(u0)

    xarr.append(curx)
    thetaarr.append(curtheta)
    farr.append(curf)

from animate_cartpole import animate_cartpole
animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=L, save_gif=True, name='cartpole_mjpc.gif')






