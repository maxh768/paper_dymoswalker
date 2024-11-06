import casadi as ca
import numpy as np

# Define the system dynamics: x_{k+1} = A x_k + B u_k
A = np.array([[ 0.763,  0.460,  0.115,  0.020],
              [-0.899,  0.763,  0.420,  0.115],
              [ 0.115,  0.020,  0.763,  0.460],
              [ 0.420,  0.115, -0.899,  0.763]])
B = np.array([[0.014],
              [0.063],
              [0.221],
              [0.367]])
Q = np.eye(4)                   # State cost
R = np.eye(1)                   # Control cost
R[0,0] = 1e-8

# Define the prediction horizon
N = 25  # Prediction horizon

# Define the objective and constraints
# The objective is to minimize state and control costs
objective = 0

# Initial state
x0 = np.matrix([[3.0], [5.0], [0.0], [0.0]])

# CasADi optimization setup
opti = ca.Opti()

# Define optimization variables for states and controls over the horizon
X = opti.variable(4, N+1)  # States for each time step
U = opti.variable(1, N)    # Control inputs for each time step

# Initial state constraint
opti.subject_to(X[:, 0] == x0)

# Loop to define the dynamics, objective, and constraints
opti.minimize((X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2))
for k in range(N):
    # Dynamics constraint: x_{k+1} = A x_k + B u_k
    opti.subject_to(X[:, k+1] == ca.mtimes(A, X[:, k]) + ca.mtimes(B, U[:, k]))
    
    # Add the state and control cost to the objective
    objective += ca.mtimes(X[:, k].T, ca.mtimes(Q, X[:, k])) + ca.mtimes(U[:, k].T, ca.mtimes(R, U[:, k]))

# Set the objective
opti.minimize((X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2))

# Add input constraints (e.g., control bounds)
u_min, u_max = -.50, .50  # Control input limits
opti.subject_to(opti.bounded(u_min, U, u_max))

# Create the solver
#opts = {'ipopt.print_level': 0, 'print_time': 0}
#solver_opts = {'opts': opts}
opti.solver('ipopt')


x1arr = []
x2arr = []
dx1arr = []
dx2arr = []
tarr = []
uarr = []
carr = []
# Simulation loop
x_current = x0
for i in range(25):  # Simulate for 50 time steps
    # Solve the optimization problem
    sol = opti.solve()

    x1 = float(x_current[0])
    x2 = float(x_current[1])
    x3 = float(x_current[2])
    x4 = float(x_current[3])
    x1arr = np.append(x1arr, x1)
    x2arr = np.append(x2arr, x2)
    dx1arr = np.append(dx1arr, x3)
    dx2arr = np.append(dx2arr, x4)


    # Extract the optimal control input for the first time step
    u_opt = sol.value(U[:, 0])
    x_opt = sol.value(X)
    #print(x_opt)
    J = sol.value(objective)

    uarr = np.append(uarr, u_opt)
    tarr = np.append(tarr, i)
    carr = np.append(carr, J)
    

    # Apply the control and update the state
    x_current = A @ x_current + B * u_opt

    print(f"Step {i}: x = {x_current}, u = {u_opt}")
    print(x_opt)

    

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3)

# position states
ax1.plot(tarr, x1arr, label='X1')
ax1.plot(tarr, x2arr, label='X2')
ax1.plot(tarr, dx1arr, label='X3')
ax1.plot(tarr, dx2arr, label='X4')
ax1.legend()

# position states
ax2.plot(tarr, uarr, label='U')
ax2.legend()

# position states
ax3.plot(tarr, carr, label='Cost')
ax3.legend()

plt.savefig('osciliating.jpg', bbox_inches='tight')
