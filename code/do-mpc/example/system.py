import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

"""
## CONFIG SYSTEM
"""

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)


# set states
x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))

dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))


# control / inputs ??
#x1_des = model.set_variable(var_type='_u', var_name='x1_des', shape=(1,1))
#x2_des = model.set_variable(var_type='_u', var_name='x2_des', shape=(1,1))
tau = model.set_variable(var_type='_u', var_name='tau', shape=(1,1))

#print('x1={}, with x1.shape={}'.format(x1, x1.shape))
#print('x2={}, with x2.shape={}'.format(x2, x1.shape))
#print('dx1={}, with dx1.shape={}'.format(dx1, dx1.shape))
#print('dx2={}, with dx2.shape={}'.format(dx2, dx2.shape))

# set params
a = 0.5
b = 0.5
mh = 10
m = 5
phi = 0.05
l = a + b
g = 9.81

# set rhs
model.set_rhs('x1',dx1)
model.set_rhs('x2',dx2)

H22 = (mh + m)*(l**2) + m*a**2
H12 = -m*l*b*np.cos(x2 - x1)
H11 = m*b**2
h = -m*l*b*np.sin(x1-x2)
G2 = -(mh*l + m*a + m*l)*g*np.sin(x2)
G1 = m*b*g*np.sin(x1)
K = 1 / (H11*H22 - (H12**2)) # inverse constant
dx1set = (H12*K*h*dx1**2) + (H22*K*h*dx2**2) - H22*K*G1 + H12*K*G2 - (H22 + H12)*K*tau
dx2set = (-H11*K*h*dx1**2) - (H12*K*h*dx2**2) + H12*K*G1 - H11*K*G2 + ((H12 + H11)*K*tau)

model.set_rhs('dx1',dx1set)
model.set_rhs('dx2',dx2set)

model.setup()

"""
## CONFIG CONTROLLER
"""

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 0,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# obj function
mterm = x1**2 + x2**2
lterm = x1**1 + x2**2
mpc.set_objective(mterm=mterm, lterm=lterm)

# set r term ??
mpc.set_rterm(
    tau=1e-2
)

# lower and upper bounds on states
mpc.bounds['lower','_x','x1'] = -1.5708 # -90 deg
mpc.bounds['lower','_x','x2'] = -1.5708 # -90 deg
mpc.bounds['upper','_x','x1'] = 1.5708 # +90 deg
mpc.bounds['upper','_x','x2'] = 1.5708 # +90 deg\

# lower and upper bounds on inputs (tau/desired pos?)
mpc.bounds['lower','_u','tau'] = -10
mpc.bounds['upper','_u','tau'] = 10

# should maybe add scaling to adjust for difference in magnitude from diferent states (optinal/future)

# set uncertain parameters (none for now)

mpc.setup()

"""
## CONFIG SIMULATOR
"""

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = 0.1)

# uncertain vars (future)

simulator.setup()

"""
## CONTROL LOOP
"""

# initial guess
x0 = np.array([-0.3, -0.4, 0.2, -1.05]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

print(mpc.x0['x1'])
print(mpc.x0['x2'])
print(mpc.x0['dx1'])
print(mpc.x0['dx2'])

# graphics
import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)


# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='x1', axis=ax[0])
    g.add_line(var_type='_x', var_name='x2', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='tau', axis=ax[1])


ax[0].set_ylabel('angle position [rad]')
ax[1].set_ylabel('torque [N*m]')
ax[1].set_xlabel('time [s]')


## natural responce of system (needs collision events to be added in sys dynamics)
u0 = np.zeros((1,1))
for i in range(200):
    simulator.make_step(u0)
sim_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
sim_graphics.reset_axes()
# Show the figure:
fig.savefig('fig_runsimulator.png')

# run optimizer
u0 = mpc.make_step(x0)
sim_graphics.clear()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()
# Show the figure:
fig.savefig('fig_runopt.png')


# finish running control loop
simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()


for i in range(200):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
fig.savefig('mainloop.png')