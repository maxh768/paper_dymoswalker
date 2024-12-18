from scipy import sparse
import osqp
import mujoco
import glfw
import numpy as np
from numpy.linalg import inv

from mj_interface import mjmod_init, mjrend_init, linearize, setpolelen
from model import model_set

# import opengl to save results as a mp4
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

plotting = True

# set initial conditions
x0 = [0, np.deg2rad(180)]
model, data = mjmod_init(x0)

window, camera, scene, context, viewport = mjrend_init(model, data)

# do osqp set up
A, B = linearize(model, data)
Ad = sparse.csc_matrix(A)
Bd = sparse.csc_matrix(B)
[nx, nu] = Bd.shape
# Constraints
u0 = 0
umin = np.array([-25])
umax = np.array([25])
xmin = np.array([-2.5, -100, -100, -100])
xmax = np.array([2.5, 100, 100, 100])

# Objective function
Q = sparse.diags([1, 5, 1, 1])
QN = Q
R = 0.01*sparse.eye(4)

# Initial and reference states
x0 = np.array([0, np.deg2rad(180), 0, 0])
xr = np.zeros(4)

N = 20 # horizon

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective

q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nx)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

print('P shape:', P.shape)
print('q shape:', q.shape)
print('A shape:', A.shape)
print('l shape:', l.shape)
print('u shape:', u.shape)
print('P', P)
print('q', q)
print('A', A)
print('l', l)
print('u', u)

# Setup workspace
prob.setup(P, q, A, l, u)


# set up recording
if plotting:
    width, height = glfw.get_framebuffer_size(window)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('cartpole.mp4', fourcc, 30.0, (width, height))

stop = False
Fail = False
step = 1
while(stop==False):
    if step != 1:
        if glfw.window_should_close(window): # if window is closed
            stop = True


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

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    print(ctrl)
    

    # mj step2: run with ctrl input
    mujoco.mj_step2(model, data)




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
