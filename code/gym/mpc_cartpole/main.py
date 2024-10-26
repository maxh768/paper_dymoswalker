from scipy import sparse
import osqp
import mujoco
import glfw
import numpy as np
from numpy.linalg import inv

"""
Set up rendering and mujoco model
"""
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

model = mujoco.MjModel.from_xml_path('/home/max/workspace/research_template/code/gym/inverted_pendulum.xml')
data = mujoco.MjData(model)
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


"""
set initial state
"""
data.qpos = np.array([0, np.deg2rad(0)])
mujoco.mj_forward(model, data)


"""
set variables to hold mujoco parameters
"""
nq = model.nq # num of position coords
nv = model.nv # num DOF
nc = data.nefc # num of active constraints SIZE CHANGES, may cause issues

q = np.zeros(nq) # joint positions
v = np.zeros(nv) # joint velocities
T = np.zeros(nv) # total applied force (passive + actuation + external)
c = np.zeros(nv) # bias force (coriolis + centrifugal + gravitational)
M = np.zeros((nv,nv)) # inertia matrix
J = np.zeros((nc,nv)) # uses nc, may be some issues here
f = np.zeros(nc) # also uses nc, could cause some issues

gear_ratio = model.actuator_gear # data.qfrc_actuator = gear_ratio * data.ctrl
gear_ratio = gear_ratio[:,0]


"""
setup osqp MPC problem
"""


while(not glfw.window_should_close(window)):

    """
    mj step1: calculate generalized positions and velocities in mujoco model
    """
    mujoco.mj_step1(model, data)

    """
    control steps here:
    1a. Maybe udpate x0 here, not sure if it should be before/after mjstep1
    1. Update Model (Calculate J,M,etc from mujoco)
    2. Solve OSQP MPC Problem
    3 . Apply control input to model
    """

    # update model
    # get matrix sizes
    nq = model.nq # num of position coords
    nv = model.nv # num DOF
    nc = data.nefc # num of active constraints SIZE CHANGES, may cause issues
    
    # get muj matrices
    mujoco.mj_fullM(model, M, data.qM) # ---> nv x nv (already size correct)
    c = data.qfrc_bias # ---> nv (already size correct)
    f = data.qfrc_constraint # ---> nc
    T = data.qfrc_actuator + data.qfrc_passive + data.qfrc_applied # ---> nv (already size correct)
    J = data.efc_J # ---> nc x nv
    #data.ctrl = 0.2
    q = data.qpos
    print(q)
    #print(f)
    # adjust matrix sizes to real size
    f = f[nc:]
    #print(f)
    #print(J)
    J = J[0:nc,0:nv]
    #print(J)
    if nc == 0:
        Jtf = np.zeros(nv)
    else:
        Jtf = np.transpose(J).dot(f)
    #print('T: ',T)
    #print('Jtf: ', Jtf)
    #print('c: ',c)
    #print('Addadition: ', (T + Jtf - c))
    checkacc = inv(M).dot((T + Jtf - c))
    realacc = data.qacc

    """I = np.identity(nv)
    Aeq = np.block([[-I,M]])
    beq = -c

    q_pred = Aeq*q
    print(q_pred)"""








    """
    mj step 2: calculate forces and acceleration based on control input
    """
    mujoco.mj_step2(model, data)

    # get calculcated acceleration to compare
    #realacc = data.qacc
    #print(realacc)

    """
    Not sure: update x0 here or during control steps...
    """

    """
    render mujoco frames
    """
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

# close window
glfw.terminate()