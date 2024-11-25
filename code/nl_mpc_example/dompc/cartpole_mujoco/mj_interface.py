import mujoco
import glfw
import numpy as np
from numpy.linalg import inv
import copy
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *



# initialize the model
def mjmod_init(x0):

    path2xml = '/home/max/workspace/research_template/code/nl_mpc_example/dompc/cartpole_mujoco/inverted_pendulum.xml'
    model = mujoco.MjModel.from_xml_path(path2xml)
    data = mujoco.MjData(model)


    # initial condition (remove?)
    data.qpos = np.array([x0[0], x0[1]])
    mujoco.mj_forward(model, data)

    return model, data

# init window to render
def mjrend_init(model, data):
    np.set_printoptions(precision=4)

    def init_window(max_width, max_height):
        glfw.init()
        window = glfw.create_window(width=max_width, height=max_height,
                                        title='Demo', monitor=None,
                                        share=None)
        glfw.make_context_current(window)
        return window

    window = init_window(2400, 1800)
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    scene = mujoco.MjvScene(model, 6000)
    camera = mujoco.MjvCamera()
    camera.trackbodyid = 2
    camera.distance = 3
    camera.azimuth = 90
    camera.elevation = -20
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    return window, camera, scene, context, viewport


# wrapper to clone model for lineariztaion
def clone_mj(model, data):
    cmod = copy.deepcopy(model)
    cdat = copy.deepcopy(data)
    return cmod, cdat


# forcing term for linearization
def f(x, u, model, data):
    #x=q0,q1,qdot0,qdot1
    #u=torque
    data.qpos[0] = x[0]
    data.qpos[1] = x[1]
    data.qvel[0] = x[2]
    data.qvel[1] = x[3]
    data.ctrl = u
    mujoco.mj_forward(model,data)

    #qddot = inv(M)*(data_ctrl-frc_bias)
    M = np.zeros((2,2))
    mujoco.mj_fullM(model,M,data.qM)
    c = data.qfrc_bias
    T = data.qfrc_actuator + data.qfrc_passive + data.qfrc_applied
    taucheck = data.qfrc_passive + data.qfrc_actuator + data.qfrc_applied
    #print('mj tau: ', )
    #print('qfrc_pass: ',data.qfrc_passive)
    #print('used tau: ', 30*np.array([data.ctrl[0]]))

    qddot = inv(M).dot((T - c))

    xdot = np.array([data.qvel[0],data.qvel[1],qddot[0],qddot[1]])

    return xdot

# linearize model about current point
def linearize(model, data):

    n = 4
    m = 1
    A = np.zeros((n,n))
    B = np.zeros((n,m))
    x0 = np.zeros(4)

    x0[0] = data.qpos[0]
    x0[1] = data.qpos[1]
    x0[2] = data.qvel[0]
    x0[3] = data.qvel[1]
    u0 = data.ctrl
    cmod, cdat = clone_mj(model, data)
    xdot0 = f(x0, u0, cmod, cdat)
    #print('Calcd: ', xdot0)

    pert = 1e-2
    #get A matrix
    for i in range(0,n):
        x = [0]*n
        u = u0
        for j in range(0,n):
            x[j] = x0[j]
        x[i] = x[i]+pert
        cmod, cdat = clone_mj(model, data)
        xdot = f(x,u, cmod, cdat)
        for k in range(0,n):
            A[k,i] = (xdot[k]-xdot0[k])/pert

    #get B matrix
    for i in range(0,m):
        x = x0
        u = [0]*m
        for j in range(0,m):
            u[j] = u0[j]
        u[i] = u[i]+pert
        cmod, cdat = clone_mj(model, data)
        xdot = f(x,u, cmod, cdat)
        for k in range(0,n):
            B[k,i] = (xdot[k]-xdot0[k])/pert

    return A,B


