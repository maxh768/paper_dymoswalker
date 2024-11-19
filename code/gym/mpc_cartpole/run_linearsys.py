from scipy import sparse
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

path2xml = '/home/max/workspace/research_template/code/gym/mpc_cartpole/inverted_pendulum.xml'
model = mujoco.MjModel.from_xml_path(path2xml)
data = mujoco.MjData(model)
options = model.opt
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




x = np.zeros(4)
dx_real = np.zeros(4)
from linearize_system import linearize
while(not glfw.window_should_close(window)):

    """
    mj step1: calculate generalized positions and velocities in mujoco model
    """
    mujoco.mj_step1(model, data)

    A, B = linearize(model, data)

    x[0] = data.qpos[0]
    x[1] = data.qpos[1]
    x[2] = data.qvel[0]
    x[3] = data.qvel[1]
    u = data.ctrl
    dx_check = A.dot(x) + B.dot(u)


    data.ctrl = 0.3
    #print(A)
    #print(B)
    #print('Calcd: ', dx_check)
    



    mujoco.mj_step2(model, data)
    dx_real[0] = data.qvel[0]
    dx_real[1] = data.qvel[1]
    dx_real[2] = data.qacc[0]
    dx_real[3] = data.qacc[1]
    print('Real: ', dx_real)
    


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