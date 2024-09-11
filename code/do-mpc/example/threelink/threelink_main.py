import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

# set simulation parameters
num_steps = 100
delta_t = 0.01

#import unlocked
from sys_unlocked import model_unlocked
model_unlocked = model_unlocked()
from unlocked_controller import control_unlocked
mpc_unlocked = control_unlocked(model_unlocked, delta_t=delta_t)

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
## CONFIG BOTH SIMULATORs
"""
#unlocked sim
simulator_unlocked = do_mpc.simulator.Simulator(model_unlocked)
simulator_unlocked.set_param(t_step = delta_t)
# uncertain vars (future)
simulator_unlocked.setup()

#locked sim
simulator_locked = do_mpc.simulator.Simulator(model_locked)
simulator_locked.set_param(t_step = delta_t)
# uncertain vars (future)
simulator_locked.setup()


"""
CONTROL LOOP
"""

x10 = 0.1877
x20 = -0.2884
x30 = -0.2884
x40 = -1.1014
x50 = -0.0399
x60 = -0.0399
# initial guess
x0 = np.array([x10, x20, x30, x40, x50, x60]).reshape(-1,1)
simulator_unlocked.x0 = x0
mpc_unlocked.x0 = x0
mpc_unlocked.set_initial_guess()


# graphics
import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc_unlocked.data)
sim_graphics = do_mpc.graphics.Graphics(simulator_unlocked.data)


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
simulator_unlocked.reset_history()
simulator_unlocked.x0 = x0
mpc_unlocked.reset_history()

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


"""
MAIN LOOP
"""
#curx6 = float(mpc_unlocked.x0['dx3',0])

num_locked = 0
marker = 0
phibound = [1,1]
kneelock = False
stop = False
from threelink_trans import kneestrike, heelstrike
for i in range(num_steps):
    stepnum = i+1
    curx1 = x0[0,0]
    curx2 = x0[1,0]
    curx3 = x0[2,0]
    curx4 = x0[3,0]
    curx5 = x0[4,0]
    curx6 = x0[5,0]
    #print(x0)
    print([curx1, curx2, curx3, curx4, curx5, curx6])
    print(stepnum)

    u0 = mpc_unlocked.make_step(x0)
    x0 = simulator_unlocked.make_step(u0)

    if (curx2-curx3 < 0) and (stepnum-marker > 5):
        print('KNEESTRIKE')
        print('before kneestrike: ', [curx1, curx2, curx3, curx4, curx5, curx6])
        marker = stepnum
        kneelock = True
        #knee strike
        newstates = kneestrike(curx1, curx2, curx3, curx4, curx5, curx6, a1=a1, a2=a2, b1=b1, b2=b2, mh=mh, mt=mt, ms=ms)
        print('after kneestrike: ', newstates)
        x0 = np.array([newstates[0], newstates[1], newstates[2], newstates[3]]).reshape(-1,1)
        mpc_locked.x0 = x0
        mpc_locked.set_initial_guess()
        simulator_locked.x0 = x0

        while(kneelock==True):
        #for k in range(50):
            num_locked = num_locked+1
            curx1= x0[0,0]
            curx2= x0[1,0]
            curx3= x0[2,0]
            curx4 = x0[3,0]

            phibound[0] = phibound[1]
            phibound[1] = curx1+ curx2
            #print(curx1)
            #print(curx2)
            print(num_locked)
            print(phibound[1])
            if (((phibound[0] > -0.1) and (phibound[1] < -0.1)) or ((phibound[0] <-0.1) and (phibound[1] > -0.1))):
                print('HEELSTRIKE')
                newstates = heelstrike(curx1, curx2, curx3, curx4, a1=a1, a2=a2, b1=b1, b2=b2, mh=mh, mt=mt, ms=ms)
                print('after heelstrike: ', newstates)
                print(newstates)
                x0 = np.array([newstates[0], newstates[1], newstates[2], newstates[3], newstates[4], newstates[5]]).reshape(-1,1)
                simulator_unlocked.x0 = x0
                num_locked = 0
                kneelock = False
            
            if kneelock == True:
                u0 = mpc_locked.make_step(x0)
                x0 = simulator_locked.make_step(u0)
            if num_locked > 100:
                stop = True
                break
        if stop==True:
            break
      


# Plot predictions from t=0
#mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
#sim_graphics.plot_results()
#sim_graphics.reset_axes()
threeleg_dir = './research_template/threeleg_graphs/'
#fig.savefig(threeleg_dir + 'mainloop.png')

## SAVE RESULTS
from do_mpc.data import save_results, load_results
save_results([mpc_unlocked, simulator_unlocked])
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
animate_threelink(x1_result, x2_result,x3_result, a1, b1, a2, b2, phi, saveFig=True, gif_fps=18, name=threeleg_dir+'threeleg.gif')

os.remove('./results/results.pkl')
save_results([mpc_locked, simulator_locked])
results = load_results('./results/results.pkl')
x = results['mpc']['_x']
#print(x)
x1_result = x[:,0]
x2_result = x[:,1]

from animate import animate_compass
animate_compass(x2_result, x1_result, L/2, L/2, phi, saveFig=True, gif_fps=18)


# animate the plot window to show real time predictions and trajectory
"""from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib import animation
def update(t_ind):
    sim_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()
anim = FuncAnimation(fig, update, frames=num_steps, repeat=False)
anim.save(threeleg_dir + 'states.gif', writer=animation.PillowWriter(fps=15))"""
