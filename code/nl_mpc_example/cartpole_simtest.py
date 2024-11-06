import numpy as np
from cartpole_sys import discretize_sys
from animate_cartpole import animate_cartpole

# initial condition
x0 = 0
theta0 = np.deg2rad(170)
dx0 = 0
dtheta0 = np.deg2rad(0)
X = np.matrix([[x0], [theta0], [dx0], [dtheta0]])

# parameters
h = 0.0005
t_final = 5
t_steps = int(t_final/h)
# set cartpole parameters....


# dummy force
F = 1

# record for animation
xarr = []
thetaarr = []
farr = []

for i in range(t_steps):
    x = float(X[0])
    theta = float(X[1])
    dx = float(X[2])
    dtheta = float(X[3])
    if (i) % 200 == 0:
        print('arr')
        xarr = np.append(xarr, x)
        thetaarr = np.append(thetaarr, theta)
        farr = np.append(farr, F)
    A, B, C = discretize_sys(theta, dtheta, h)
    X = A.dot(X) + B*F + C
    print(X)

animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=0.5)