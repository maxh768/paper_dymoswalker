import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

# set simulation parameters
num_steps = 4000
delta_t = 0.001

#import locked
from sys_locked import model_locked
model_locked = model_locked()
from locked_controller import control_locked
mpc_locked = control_locked(model_locked, delta_t=delta_t)

# set params from model 
a1 = 0.375
b1 = 0.125
a2 = 0.175
b2 = 0.325
mh = 0.5
mt = 0.5
ms = 0.05
g=9.81
L = a1+b1+a2+b2
ls = a1+b1
lt = a2+b2
phi = .05

"""
## CONFIG SIMULATOR
"""

#locked sim
simulator_locked = do_mpc.simulator.Simulator(model_locked)
simulator_locked.set_param(t_step = delta_t)
# uncertain vars (future)
simulator_locked.setup()

"""INITIAL GUESS"""
#x10 = -0.1
#x20 = 0.25
#x30 = -1
#x40 = 0.6

# initial guess
x0 = np.array([-0.3, 0.2038, -0.41215, -1.05]).reshape(-1,1)
simulator_locked.x0 = x0
mpc_locked.x0 = x0
mpc_locked.set_initial_guess()


# graphics
import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc_locked.data)
sim_graphics = do_mpc.graphics.Graphics(simulator_locked.data)


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


x1_result = []
x2_result = []
from calc_transition import calc_trans
phibound = [1,1]
## natural responce of system (needs collision events to be added in sys dynamics)
u0 = np.zeros((1,1))
for i in range(num_steps):
    x0 = simulator_locked.make_step(u0)
    print(i+1)
    #print(x0)
    curx1 = x0[0]
    curx2 = x0[1]
    curx3 = x0[2]
    curx4 = x0[3]
    phibound[0] = phibound[1]
    phibound[1] = curx1 + curx2
    if ((((phibound[0] > -0.1) and (phibound[1] < -0.1)) or ((phibound[0] <-0.1) and (phibound[1] > -0.1))) and curx1>0) and (i+1)>5:
        print('TRANSITION')
        newstates = calc_trans(curx1, curx2, curx3, curx4 )
        x0 = np.array([newstates[0], newstates[1], newstates[2], newstates[3]]).reshape(-1,1)
        simulator_locked.x0 = x0
    if (i+1) % 10 == 0:
        x1_result = np.concatenate((x1_result, curx1))
        x2_result = np.concatenate((x2_result, curx2))

sim_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
sim_graphics.reset_axes()
# Show the figure:
fig.savefig('fig_runsimulator.png')

from animate import animate_compass
animate_compass(x1_result, x2_result, L/2, L/2, phi, iter=1, saveFig=True, gif_fps=20,name='twoleg_compass.gif')



## IMPROVE GRAPH
# Change the color for the states:
for line_i in mpc_graphics.pred_lines['_x', 'x1']: line_i.set_color('#1f77b4') # blue
for line_i in mpc_graphics.pred_lines['_x', 'x2']: line_i.set_color('#ff7f0e') # orange
# Change the color for the input:
for line_i in mpc_graphics.pred_lines['_u', 'tau']: line_i.set_color('#1f77b4')

# Make all predictions transparent:
for line_i in mpc_graphics.pred_lines.full: line_i.set_alpha(0.2)

# Get line objects (note sum of lists creates a concatenated list)
lines = sim_graphics.result_lines['_x', 'x1']+sim_graphics.result_lines['_x', 'x2']
ax[0].legend(lines,'12',title='state')
# also set legend for second subplot:
lines = sim_graphics.result_lines['_u', 'tau']
ax[1].legend(lines,'1',title='tau')

