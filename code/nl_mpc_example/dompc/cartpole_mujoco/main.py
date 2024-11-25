import numpy as np
import mujoco
import glfw
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc
delta_t = .02


# import model and controller
from model import model_set
from controller import control

# initialize mujoco and renderer
from mj_interface import mjmod_init, mjrend_init
x0 = [0, np.deg2rad(180)]
model, data = mjmod_init(x0)
window, camera, scene, context, viewport = mjrend_init(model, data)

frames = []    

from mj_interface import linearize

# set matrices for plotting
xarr = []
thetaarr = []
farr = []
#jarr = []
tarr = []
yarr = []
the_dmpc = []

# start main loop
x = np.zeros(4)
step = 1
while(not glfw.window_should_close(window)):

    # mj step 1: pre control
    mujoco.mj_step1(model, data)

    # get linearized system
    A, B = linearize(model, data)
    #print(A)
    #print(B)
    # model and controller
    dmpc_mod = model_set(A, B)
    mpc = control(dmpc_mod, delta_t)

    # estimator and simulator (need to replace with mujoco)
    estimator = do_mpc.estimator.StateFeedback(dmpc_mod)
    simulator = do_mpc.simulator.Simulator(dmpc_mod)
    simulator.set_param(t_step = delta_t)
    simulator.setup()

    # get current state
    x[0] = data.qpos[0]
    x[1] = data.qpos[1]
    x[2] = data.qvel[0]
    x[3] = data.qvel[1]

    # Initial state
    mpc.x0 = x
    simulator.x0 = x
    estimator.x0 = x

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    # get control
    u = mpc.make_step(x)
    y_next = simulator.make_step(u)
    cury = y_next[0]
    curthe_dmpc = y_next[1]


    data.ctrl = u
    curf = u
    curt = delta_t*step


    # mj step2: run with ctrl input
    mujoco.mj_step2(model, data)

    curx = data.qpos[0]
    curtheta = data.qpos[1]

    # append arrays
    xarr = np.append(xarr, curx)
    thetaarr = np.append(thetaarr, curtheta)
    farr = np.append(farr, curf)
    tarr = np.append(tarr, curt)
    yarr = np.append(yarr, cury)
    the_dmpc = np.append(the_dmpc, curthe_dmpc)

    step += 1
    # render frames

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

# close window
glfw.terminate()


# plotting
import matplotlib.pyplot as plt
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
plt.savefig('cartpole_mjpc_times', bbox_inches='tight')

thetaarr = thetaarr - np.pi
from animate_cartpole import animate_cartpole
animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=1, save_gif=True, name='cartpole_mjpc.gif')

