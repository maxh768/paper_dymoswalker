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

# set number of steps
num_steps = 380
delta_t = 0.01

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)


#set states
x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))

dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))



# control / inputs
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

# dynamics
H22 = (mh + m)*(l**2) + m*a**2
H12 = -m*l*b*np.cos(x2 - x1)
H11 = m*b**2
h = -m*l*b*np.sin(x1-x2)
G2 = -(mh*l + m*a + m*l)*g*np.sin(x2)
G1 = m*b*g*np.sin(x1)

K = 1 / (H11*H22 - (H12**2)) # inverse constant
dx1set = (H12*K*h*dx1**2) + (H22*K*h*dx2**2) - H22*K*G1 + H12*K*G2 - (H22 + H12)*K*tau
dx2set = (-H11*K*h*dx1**2) - (H12*K*h*dx2**2) + H12*K*G1 - H11*K*G2 + ((H12 + H11)*K*tau)


# set rhs
model.set_rhs('x1',dx1)
model.set_rhs('x2',dx2)
model.set_rhs('dx1',dx1set)
model.set_rhs('dx2',dx2set)


model.setup()

"""
## CONFIG CONTROLLER
"""

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 70,
    't_step': delta_t,
    'n_robust': 1,
    'store_full_solution': True,
    #'supress_ipopt_output': True
}
mpc.settings.supress_ipopt_output()
mpc.set_param(**setup_mpc)



# obj function
mterm = (x1-0.19)**2 + (x2+0.3)**2
lterm = (x1-0.19)**1 + (x2+0.3)**2
mpc.set_objective(mterm=mterm, lterm=lterm)

# set r term ??
mpc.set_rterm(
    tau=10
)

# lower and upper bounds on states
mpc.bounds['lower','_x','x1'] = -1.5708 # -90 deg
mpc.bounds['lower','_x','x2'] = -1.5708 # -90 deg
mpc.bounds['upper','_x','x1'] = 1.5708 # +90 deg
mpc.bounds['upper','_x','x2'] = 1.5708 # +90 deg

# lower and upper bounds on inputs (tau/desired pos?)
#mpc.bounds['lower','_u','tau'] = -3
#mpc.bounds['upper','_u','tau'] = 3

# should maybe add scaling to adjust for difference in magnitude from diferent states (optinal/future)

# set uncertain parameters (none for now)

mpc.setup()

"""
## CONFIG SIMULATOR
"""

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = delta_t)

# uncertain vars (future)

simulator.setup()

"""
## CONTROL LOOP
"""

# initial guess
x0 = np.array([-0.3, 0.2038, -0.41215, -1.05]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

#print(mpc.x0['x1'])
#print(mpc.x0['x2'])
#print(mpc.x0['dx1'])
#print(mpc.x0['dx2'])

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
x1_result = []
x2_result = []
from calc_transition import calc_trans
phibound = [1,1]
## natural responce of system (needs collision events to be added in sys dynamics)
u0 = np.zeros((1,1))
"""for i in range(num_steps):
    x0 = simulator.make_step(u0)
    print(i+1)
    #print(x0)
    curx1 = x0[0]
    curx2 = x0[1]
    curx3 = x0[2]
    curx4 = x0[3]
    phibound[0] = phibound[1]
    phibound[1] = curx1 + curx2
    if (((phibound[0] > -0.1) and (phibound[1] < -0.1)) or ((phibound[0] <-0.1) and (phibound[1] > -0.1))) and curx1>0:
        print('TRANSITION')
        newstates = calc_trans(curx1, curx2, curx3, curx4, m=m, mh=mh, a=a, b=b)
        x0 = np.array([newstates[0], newstates[1], newstates[2], newstates[3]]).reshape(-1,1)
        simulator.x0 = x0
    #if (i+1) % 10 == 0:
    x1_result = np.concatenate((x1_result, curx1))
    x2_result = np.concatenate((x2_result, curx2))

sim_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
sim_graphics.reset_axes()
# Show the figure:
fig.savefig('fig_runsimulator.png')

from animate import animate_compass
animate_compass(x1_result, x2_result, a, b, phi, iter=1, saveFig=True, gif_fps=20,name='twoleg_compass.gif')

# run optimizer
u0 = mpc.make_step(x0)
sim_graphics.clear()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()
# Show the figure:
#fig.savefig('fig_runopt.png')"""

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


# finish running control loop
simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

# delete all previous results so the animation works
import os 
# Specify the directory containing the files to be deleted 
directory = './results/' 
# Get a list of all files in the directory 
files = os.listdir(directory) 
# Loop through the files and delete each one 
for file in files: 
    file_path = os.path.join(directory, file) 
    os.remove(file_path) 

phibound = [0, 0]
# main loop
from calc_transition import calc_trans
u0 = mpc.make_step(x0)
x0 = simulator.make_step(u0)
#print(mpc.x0['x1',0])
curx1 = mpc.x0['x1',0]
curx2 = mpc.x0['x2',0]
curx3 = mpc.x0['dx1',0]
curx4 = mpc.x0['dx2',0]
numiter = 1
for i in range(num_steps-1):
    curx1 = mpc.x0['x1',0]
    curx2 = mpc.x0['x2',0]
    curx3 = mpc.x0['dx1',0]
    curx4 = mpc.x0['dx2',0]
    phibound[0] = phibound[1]
    phibound[1] = curx1 + curx2
    #print('x1: ',curx1)
    #print('x2: ',curx2)
    print('x1 (deg): ', curx1*(180/np.pi))
    print('x2 (deg): ', curx2*(180/np.pi))
    print('x1+x2: ', phibound[1])
    print('step num: ', i+2)
    if (((phibound[0] > -0.1) and (phibound[1] < -0.1)) or ((phibound[0] <-0.1) and (phibound[1] > -0.1))) and curx1>0:
        print('TRANSITION')
        newstates = calc_trans(curx1, curx2, curx3, curx4, m=m, mh=mh, a=a, b=b)
        x0 = np.array([newstates[0], newstates[1], newstates[2], newstates[3]]).reshape(-1,1)
        simulator.x0 = x0
        numiter = numiter + 1
        numpoints = i+2


    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    
twoleg_dir = './research_template/twoleg_graphs/'
# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
fig.savefig(twoleg_dir + 'twoleg_mainloop.png')

## SAVE RESULTS
from do_mpc.data import save_results, load_results
save_results([mpc, simulator])
results = load_results('./results/results.pkl')

x = results['mpc']['_x']
#print(x)
x1_result = x[:,0]
x2_result = x[:,1]
x3_result = x[:,2]
x4_result = x[:,3]

# animate motion of the compass gait
from animate import animate_compass
animate_compass(x1_result, x2_result, a, b, phi, iter=1, saveFig=True, gif_fps=20,name=twoleg_dir + 'twoleg_compass.gif')

# animate the plot window to show real time predictions and trajectory
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib import animation
def update(t_ind):
    sim_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()
anim = FuncAnimation(fig, update, frames=num_steps, repeat=False)
anim.save(twoleg_dir + 'twoleg_statesanim.gif', writer=animation.PillowWriter(fps=15))