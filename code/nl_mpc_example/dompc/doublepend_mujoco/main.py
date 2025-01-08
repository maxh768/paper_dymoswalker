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

# import model and controller
from model import model_set
from controller import control

# import mujoco interface
from mj_interface import mjmod_init, mjrend_init, linearize, setpolelen

# import opengl to save results as a mp4
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# set initial conditions
delta_t = 0.02
x0 = [np.deg2rad(180), 0]
model, data = mjmod_init(x0)


# init window
window, camera, scene, context, viewport = mjrend_init(model, data)

# start main loop
x = np.zeros(4)
step = 1
rgb = []
depth = []


# set up recording
width, height = glfw.get_framebuffer_size(window)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('doub_pend.mp4', fourcc, 30.0, (width, height))

# init arrays
tarr = [] # time
u1arr = [] # input 1
u2arr = [] # input 2

# arrays from mj
th1arr = []
th2arr = []
th1darr = []
th2darr = []

# arrays from dompc
mpc_1 = []
mpc_2 = []
mpc_d1 = []
mpc_d2 = []

stop = False
Fail = False
while(stop==False):

    if step != 1:
        if glfw.window_should_close(window): # if window is closed
            stop = True

    # record image from window and save to video:
    glReadBuffer(GL_FRONT)
    pixel_data = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
    # Convert pixel data to a numpy array
    image = np.frombuffer(pixel_data, dtype=np.uint8)
    image = image.reshape(height, width, 3)
    # Flip the image vertically (OpenGL's origin is at the bottom left)
    image = np.flipud(image)
    video_writer.write(image)

    # mj step 1: pre control
    mujoco.mj_step1(model, data)

    # get linearized system
    A, B = linearize(model, data)
    print(A)
    print(B)
    # model and controller
    dmpc_mod = model_set(A, B)
    mpc = control(dmpc_mod, delta_t)

    # estimator and simulator (need to replace with mujoco)
    estimator = do_mpc.estimator.StateFeedback(dmpc_mod)
    simulator = do_mpc.simulator.Simulator(dmpc_mod)
    simulator.set_param(t_step = delta_t)
    simulator.setup()

    # Initial state
    x[0] = data.qpos[0]
    x[1] = data.qpos[1]
    x[2] = data.qvel[0]
    x[3] = data.qvel[1]
    mpc.x0 = x
    simulator.x0 = x
    estimator.x0 = x

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    # get control
    u = mpc.make_step(x)
    y_next = simulator.make_step(u)

    data.ctrl[0] = u[0]
    data.ctrl[1] = u[1]

    curt = delta_t*step
    tarr.append(curt)

    u1arr.append(u[0])
    u2arr.append(u[1])

    mpc_1.append(y_next[0])
    mpc_2.append(y_next[1])
    mpc_d1.append(y_next[2])
    mpc_d2.append(y_next[3])


    # mj step2: run with ctrl input
    mujoco.mj_step2(model, data)

    th1arr.append(data.qpos[0])
    th2arr.append(data.qpos[1])
    th1darr.append(data.qvel[0])
    th2darr.append(data.qvel[1])

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

# save video
video_writer.release()

# plot timeseries
from plot_results import pl_ts
pl_ts(th1arr, th2arr, th1darr, th2darr, mpc_1, mpc_2, mpc_d1, mpc_d2, u1arr, u2arr, tarr)

