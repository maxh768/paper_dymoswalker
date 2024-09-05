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
num_steps = 50
delta_t = 0.1


model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)


#set states
x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))
x3 = model.set_variable(var_type='_x', var_name='x3',shape=(1,1))

dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))
dx3 = model.set_variable(var_type='_x', var_name='dx3', shape=(1,1))



# control / inputs
tau = model.set_variable(var_type='_u', var_name='tau', shape=(1,1))



#print('x1={}, with x1.shape={}'.format(x1, x1.shape))
#print('x2={}, with x2.shape={}'.format(x2, x1.shape))
#print('dx1={}, with dx1.shape={}'.format(dx1, dx1.shape))
#print('dx2={}, with dx2.shape={}'.format(dx2, dx2.shape))

# set params
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


# dynamic matrix (will be inverted)
H11 = ms*a1**2 + mt*(ls+a2)**2 + (mh+ms+mt)*L**2
H12 = -(mt*b2 + ms*lt)*L*np.cos(x2-x1)
H13 = -ms*b1*L*np.cos(x3-x1)
H22 = mt*b2**2 + ms*lt**2
H23 = ms*lt*b1*np.cos(x3-x2)
H33 = ms*b1**2
H21 = H12
H31 = H13
H32 = H23

# B matrix
B11 = 0
B12 = (-(mt*b2+ms*lt)*L*np.sin(x1-x2))*dx2
B13 = (-ms*b1*L*np.sin(x1-x3))*dx3
B21 = ((mt*b2+ms*lt)*L*np.sin(x1-x2))*dx1
B22 = 0
B23 = (ms*lt*b1*np.sin(x3-x2))*dx3
B31 = (ms*b1*L*np.sin(x1-x3))*dx1
B32 = (-ms*lt*b1*np.sin(x3-x2))*dx2
B33 = 0

# Gravity Matrix
G1 = -(ms*a1+mt*(ls+a2)+(mh+ms+mt)*L)*g*np.sin(x1)
G2 = (mt*b2 + ms*lt)*g*np.sin(x2)
G3 = (ms*b1*g*np.sin(x3))

# inverting H matrix
#determinate
detH = H11*(H33*H22 - H23**2) - H21*(H33*H12 - H23*H13) + H13*(H23*H12 - H22*H13)
# inverse terms
H_I11 = (H33*H22 - H23**2)/detH
H_I12 = (H13*H23 - H33*H12)/detH
H_I13 = (H12*H23 - H13*H22)/detH
H_I22 = (H33*H11 - H13**2)/detH
H_I23 = (H12*H13 - H11*H23)/detH
H_I33 = (H11*H22 - H12**2)/detH
H_I21 = H_I12
H_I31 = H_I13
H_I32 = H_I23

# U matrix (torque)
U1 = 0
U2 = 0
U3 = 0

dx1set = -( (H_I12*B21 + H_I13*B31)*x1 + (H_I11*B12 + H_I13*B32)*x2 + (H_I11*B13 + H_I12*B23)*x2 ) - (H_I11*G1 + H_I12*G2 + H_I13*G3) + (H_I11*U1 + H_I12*U2 + H_I13*U3)*tau
dx2set = -( (H_I22*B21 + H_I23*B31)*x1 + (H_I21*B12 + H_I23*B32)*x2 + (H_I21*B13 + H_I22*B23)*x3 ) - (H_I21*G1 + H_I22*G2 + H_I23*G3) + (H_I21*U1 + H_I22*U2 + H_I23*U3)*tau
dx3set = -( (H_I32*B21 + H_I33*B31)*x1 + (H_I31*B12 + H_I33*B32)*x2 + (H_I31*B13 + H_I32*B23)*x3 ) - (H_I31*G1 + H_I32*G2 + H_I33*G3) + (H_I31*U1 + H_I32*U2 + H_I33*U3)*tau

# set rhs
model.set_rhs('x1',dx1)
model.set_rhs('x2',dx2)
model.set_rhs('x3',dx3)
model.set_rhs('dx1',dx1set)
model.set_rhs('dx2',dx2set)
model.set_rhs('dx3',dx3set)


model.setup()

"""
## CONFIG CONTROLLER
"""

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 25,
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
mpc.bounds['upper','_x','x3'] = 1.5708 # +90 deg
mpc.bounds['lower','_x','x3'] = -1.5708 # +90 deg

# lower and upper bounds on inputs (tau/desired pos?)
mpc.bounds['lower','_u','tau'] = -3
mpc.bounds['upper','_u','tau'] = 3

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
x10 = -0.3
x20 = 0.2038
x30 = 0.2038
x40 = -0.41215
x50 = -1.05
x60 = -1.05
# initial guess
x0 = np.array([x10, x20, x30, x40, x50, x60]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

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
    g.add_line(var_type='_x', var_name='x3', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='tau', axis=ax[1])


ax[0].set_ylabel('angle position [rad]')
ax[1].set_ylabel('torque [N*m]')
ax[1].set_xlabel('time [s]')


## natural responce of system (needs collision events to be added in sys dynamics)
u0 = np.zeros((1,1))
for i in range(num_steps):
    simulator.make_step(u0)
sim_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
sim_graphics.reset_axes()
# Show the figure:
#fig.savefig('fig_runsimulator.png')

# run optimizer
u0 = mpc.make_step(x0)
sim_graphics.clear()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()
# Show the figure:
#fig.savefig('fig_runopt.png')

## IMPROVE GRAPH
# Change the color for the states:
for line_i in mpc_graphics.pred_lines['_x', 'x1']: line_i.set_color('#1f77b4') # blue
for line_i in mpc_graphics.pred_lines['_x', 'x2']: line_i.set_color('#ff7f0e') # orange
for line_i in mpc_graphics.pred_lines['_x', 'x3']: line_i.set_color('#ff0eeb') # purple
# Change the color for the input:
for line_i in mpc_graphics.pred_lines['_u', 'tau']: line_i.set_color('#1f77b4')

# Make all predictions transparent:
for line_i in mpc_graphics.pred_lines.full: line_i.set_alpha(0.2)

# Get line objects (note sum of lists creates a concatenated list)
lines = sim_graphics.result_lines['_x', 'x1']+sim_graphics.result_lines['_x', 'x2']+sim_graphics.result_lines['_x','x3']
ax[0].legend(lines,'123',title='state')
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
"""# main loop"""
from calc_transition import calc_trans
"""u0 = mpc.make_step(x0)
x0 = simulator.make_step(u0)
#print(mpc.x0['x1',0])
curx1 = mpc.x0['x1',0]
curx2 = mpc.x0['x2',0]
curx3 = mpc.x0['dx1',0]
curx4 = mpc.x0['dx2',0]
numiter = 1"""
for i in range(num_steps):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)

    """curx1 = mpc.x0['x1',0]
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
    x0 = simulator.make_step(u0)"""
    

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
threeleg_dir = './research_template/threeleg_graphs/'
fig.savefig(threeleg_dir + 'mainloop.png')

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
x5_result = x[:,4]
x6_result = x[:,5]

# animate motion of the compass gait
from animate_threelink import animate_threelink
animate_threelink(x1_result, x2_result,x3_result, a1, b1, a2, b2, phi, saveFig=True, gif_fps=10, name=threeleg_dir+'threeleg.gif')

# animate the plot window to show real time predictions and trajectory
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib import animation
def update(t_ind):
    sim_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()
anim = FuncAnimation(fig, update, frames=num_steps, repeat=False)
anim.save(threeleg_dir + 'states.gif', writer=animation.PillowWriter(fps=15))
