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

# add end effector position
EndEffector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "EndEffector")
x = data.site_xpos[EndEffector] # x, y, z position of end effector (y = 0 for 2D)

# ref states
xr = np.array([0, 0.6,])

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

ind = np.array([0,nv]) # empty matrix to use to convert matrices to correct size
st_jacp = np.zeros((3,nv)) # site jacobian of end effector: translational
st_jacr = np.zeros((3,nv)) # site jacobian of end effector: rotational
mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector) # calculate the site jacobians
J_old = st_jacp[ind,:] # old jacobian to be used for FD

I = np.identity(nv)

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
    
    ind = np.array([0,nv])
    x = data.site_xpos[EndEffector]
    xz_pos = x[ind]
    #print(xz_pos)

    # calculate updated jacobians and do finite difference to find dJ
    mujoco.mj_jacSite(model, data, st_jacp, st_jacr, EndEffector)
    J = st_jacp[ind,:]
    dJ = (J - J_old)/0.0005
    J_old = J

    # desired
    ref = np.array([0, 1])
    d_ref = np.zeros(2)
    dd_ref = np.zeros(2)


    # build model from matrices
    a3 = np.block([np.zeros((2, 2)), -J])
    ddy_des = dd_ref + 30*(d_ref-J@data.qvel) + 3*(ref-xz_pos)
    b3 = dJ@data.qvel - ddy_des
    mujoco.mj_fullM(model, M, data.qM) # calcualte full inertia matrix
    Bias = data.qfrc_bias # bias force (c)
    Aeq = np.block([[-I, M]])
    beq = -Bias



    Q = (a3.transpose()).dot(a3)
    q = -(a3.transpose()).dot(b3)
    P = sparse.csc_matrix(Q)
    A = sparse.csc_matrix(Aeq)

    prob = osqp.OSQP()
    prob.setup(P, q, A, beq, beq, verbose=False)
    try:
        res = prob.solve()
    except:
        pass
    #print(res.x[0])
    data.ctrl = res.x[0]

    
    

    """
    mj step 2: calculate forces and acceleration based on control input
    """
    mujoco.mj_step2(model, data)

    # get calculcated acceleration to compare
    #realacc = data.qacc
    #print(realacc)



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