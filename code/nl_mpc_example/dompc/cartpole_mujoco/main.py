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

# make function to do co design using finite difference
def balance(delta_t = 0.01, plotting = False, polelen = 0.3, kg_m_pole = 16.687):

    # set initial conditions
    x0 = [0, np.deg2rad(180)]
    model, data = mjmod_init(x0)
 
    # set pole length and update mass/intertia
    model.body_mass[2] = polelen*kg_m_pole
    print(model.body_mass)
    model.body_inertia[1,2] = (1/12)*polelen*kg_m_pole*(polelen**2)
    #model.body_mass[1] = 9

    setpolelen(model, data, polelen)


    # init window
    window, camera, scene, context, viewport = mjrend_init(model, data)
    
    # set matrices for plotting
    xarr = []
    thetaarr = []
    farr = []
    #jarr = []
    tarr = []
    yarr = []
    the_dmpc = []

    cost = 0 # initialize cost 

    # start main loop
    x = np.zeros(4)
    step = 1
    rgb = []
    depth = []

    xtol = .1
    thetatol = .05 # 3 degrees
    dxtol = 0.1
    dthetatol = .05

    # set up recording
    if plotting:
        width, height = glfw.get_framebuffer_size(window)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('cartpole.mp4', fourcc, 30.0, (width, height))

    stop = False
    Fail = False
    while(stop==False):
        if step != 1:
            if abs(curx) < xtol and abs(curtheta) < thetatol and abs(curdx) < dxtol and abs(curdtheta) < dthetatol: # if success
                stop = True
            elif glfw.window_should_close(window): # if window is closed
                stop = True
            #elif abs(curdx) < dxtol and abs(curtheta) > thetatol and abs(curx) > xtol:
               # stop = True
               # Fail = True


        if plotting:
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

        cost += u**2


        data.ctrl = u
        curf = u
        curt = delta_t*step


        # mj step2: run with ctrl input
        mujoco.mj_step2(model, data)


        curx = data.qpos[0]
        curtheta = data.qpos[1]
        curdx = data.qvel[0]
        curdtheta = data.qvel[1]
        #print(curx, curtheta, curdx, curdtheta)

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
    print(float(cost))
    if plotting:
        video_writer.release()

        # plot timeseries
        from plot_results import pl_ts
        pl_ts(xarr, tarr, thetaarr, the_dmpc, yarr, farr,name='cartpole_mjpc_times')

        # make animation
        thetaarr = thetaarr - np.pi
        from animate_cartpole import animate_cartpole
        animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=1, save_gif=True, name='cartpole_mjpc.gif')
    tf = tarr[-1]
    return float(cost), tf, Fail


if __name__ == "__main__":
    balance(polelen = 0.3,plotting=True)
