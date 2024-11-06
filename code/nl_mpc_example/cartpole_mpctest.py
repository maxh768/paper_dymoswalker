import casadi as ca
import numpy as np
from cartpole_sys import discretize_sys
from animate_cartpole import animate_cartpole

# Define the optimization variables
x = ca.MX.sym('x', 4)  # State (2D)
u = ca.MX.sym('u', 1)  # Control input (1D)

# Initial state
x_ic = 0
theta_ic = np.deg2rad(0)
dx_ic = 0
dtheta_ic = np.deg2rad(0)
x0 = np.matrix([[x_ic], [theta_ic], [dx_ic], [dtheta_ic]])

# setup system
theta = float(x0[1])
dtheta = float(x0[3])
h = 0.01
A, B, C = discretize_sys(theta, dtheta, h)

Q = np.eye(4)                   # State cost
R = np.eye(1)                   # Control cost
R[0,0] = 1e-3

# Define the system dynamics as a CasADi expression
x_next = ca.mtimes(A, x) + ca.mtimes(B, u) + C

# Define the prediction horizon
N = 20  # Prediction horizon

# Define the objective and constraints
# The objective is to minimize state and control costs
objective = 0
constraints = []

# CasADi optimization setup
opti = ca.Opti()

# Define optimization variables for states and controls over the horizon
X = opti.variable(4, N+1)  # States for each time step
U = opti.variable(1, N)    # Control inputs for each time step

x_target = np.matrix([[1], [np.deg2rad(180)], [0], [0]])
objective = ca.mtimes((X[:, 0] - x_target).T, ca.mtimes(Q, (X[:, 0] - x_target)))

# Initial state constraint
opti.subject_to(X[:, 0] == x0)


for k in range(N):
    # Dynamics constraint: x_{k+1} = A x_k + B u_k
    opti.subject_to(X[:, k+1] == ca.mtimes(A, X[:, k]) + ca.mtimes(B, U[:, k]) + C)
    
    # Add the state and control cost to the objective
    objective += ca.mtimes((X[:, k] - x_target).T, ca.mtimes(Q, (X[:, k] - x_target))) + ca.mtimes(U[:, k].T, ca.mtimes(R, U[:, k]))

# Set the objective
opti.minimize(objective)


# Add input constraints (e.g., control bounds)
u_min, u_max = -5, 5  # Control input limits
opti.subject_to(opti.bounded(u_min, U, u_max))

# solver
opti.solver('ipopt')


xarr = []
thetaarr = []
farr = []
# Simulation loop
x_current = x0
x_n = A.dot(x_current) + B*3 + C
for i in range(500):  # Simulate for 50 time steps
    # Solve the optimization problem
    sol = opti.solve()

    # Extract the optimal control input for the first time step
    u_opt = sol.value(U[:, 0])


    x = float(x_current[0])
    theta = float(x_current[1])
    dx = float(x_current[2])
    dtheta = float(x_current[3])
    if (i) % 10 == 0:
        print('arr')
        xarr = np.append(xarr, x)
        thetaarr = np.append(thetaarr, theta)
        farr = np.append(farr, u_opt)

    # Apply the control and update the state
    x_current = A.dot(x_current) + B*u_opt + C

    print(f"Step {i}: x = {x_current}, u = {u_opt}")

animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=0.5)



