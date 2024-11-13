import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

# set simulation parameters
num_steps = 672
delta_t = .001

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


# finish running control loop
simulator_unlocked.reset_history()
simulator_unlocked.x0 = x0
mpc_unlocked.reset_history()

# delete all previous results so the animation works
import os 
# Specify the directory containing the files to be deleted 
"""directory = './results/' 
# Get a list of all files in the directory 
files = os.listdir(directory) 
# Loop through the files and delete each one 
for file in files: 
    file_path = os.path.join(directory, file) 
    os.remove(file_path) """


"""
MAIN LOOP
"""
x1_result = []
x2_result = []
x3_result = []
x4_result = []
x5_result = []
x6_result = []
time_result = []
tau_result = []
len_iter = []

num_locked = 0
marker = 0
phibound = [1,1]
kneelock = False
stop = False
innertime = 0
outertime = 0
iter = 0
prev_len = 0

from threelink_trans import kneestrike, heelstrike
for i in range(num_steps):
    stepnum = i+1
    curx1 = x0[0]
    curx2 = x0[1]
    curx3 = x0[2]
    curx4 = x0[3]
    curx5 = x0[4]
    curx6 = x0[5]
    curtime = np.array([(stepnum*delta_t) + innertime])
    outertime = float(curtime)
    #print([curx1[0], curx2[0], curx3[0], curx4[0], curx5[0], curx6[0]])
    print('3 Link Step #: ', stepnum)

    u0 = mpc_unlocked.make_step(x0)
    x0 = simulator_unlocked.make_step(u0)
    curtau = u0[0]
    #print(curtau)
    if (i+1) % 10 == 0:
        x1_result = np.concatenate((x1_result, curx1))
        x2_result = np.concatenate((x2_result, curx2))
        x3_result = np.concatenate((x3_result, curx3))
        x4_result = np.concatenate((x4_result, curx4))
        x5_result = np.concatenate((x5_result, curx5))
        x6_result = np.concatenate((x6_result, curx6))
        tau_result = np.concatenate((tau_result, curtau))
        time_result = np.concatenate((time_result, curtime))

    # start inner loop
    if (curx2-curx3 < 0) and (stepnum-marker > 5):
        print('KNEESTRIKE')
        num_locked = 0
        print('before kneestrike: ', [curx1[0], curx2[0], curx3[0], curx4[0], curx5[0], curx6[0]])
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
            curx1= x0[0]
            curx2= x0[1]
            curx3= x0[2]
            curx4 = x0[3]
            curtau = u0[0]
            curtime = np.array([(num_locked*delta_t) + outertime])
            innertime = float(curtime)

            phibound[0] = phibound[1]
            phibound[1] = curx1+ curx2
            print('2 Link Step #: ', num_locked)
            if ((((phibound[0] > -0.1) and (phibound[1] < -0.1)) or ((phibound[0] <-0.1) and (phibound[1] > -0.1)))) and (num_locked>3):
                print('HEELSTRIKE')
                from calc_transition import calc_trans
                print('before heelstrike', [curx1[0], curx2[0]])
                newstates = calc_trans(curx1, curx2, curx3, curx4)
                #newstates = heelstrike(curx2, curx1, curx4, curx1, a1=a1, a2=a2, b1=b1, b2=b2, mh=10, mt=5, ms=.5)
                #print('after heelstrike: ', newstates)
                #print(newstates)
                x0 = np.array([newstates[0], newstates[1], newstates[1], newstates[2], newstates[3], newstates[3]]).reshape(-1,1)
                simulator_unlocked.x0 = x0
                kneelock = False

                
                len_x1 = int(len(x1_result))
                cur_len = len_x1 - prev_len
                prev_len = cur_len
            
                len_iter = np.append(len_iter, cur_len)
                iter += 1
                #stop = True
            
            if kneelock == True:
                u0 = mpc_locked.make_step(x0)
                x0 = simulator_locked.make_step(u0)
                if (num_locked) % 10 == 0:
                    x1_result = np.concatenate((x1_result, curx1))
                    x2_result = np.concatenate((x2_result, curx2))
                    x3_result = np.concatenate((x3_result, curx2))
                    x4_result = np.concatenate((x4_result, curx3))
                    x5_result = np.concatenate((x5_result, curx4))
                    x6_result = np.concatenate((x6_result, curx4))
                    tau_result = np.concatenate((tau_result, curtau))
                    time_result = np.concatenate((time_result, curtime))
            if num_locked > 5000:
                stop = True
                break
        # end inner loop
        if stop==True:
            break

#iter += 1
len_x1 = int(len(x1_result))
final_len = len_x1 - prev_len
len_iter = np.append(len_iter, final_len)

      


"""
DATA MANAGEMENT + PLOT RESULTS
"""

# directory to plot in
threeleg_dir = './threeleg_graphs/'


"""## SAVE RESULTS
from do_mpc.data import save_results, load_results
save_results([mpc_unlocked, simulator_unlocked])
results = load_results('./results/results.pkl')
#x = results['mpc']['_x']
#print(x)
x1_result = x[:,0]
x2_result = x[:,1]
x3_result = x[:,2]
x4_result = x[:,3]
x5_result = x[:,4]
x6_result = x[:,5]"""


# animate motion of the compass gait
from animate_threelink import animate_threelink
animate_threelink(x1_result, x2_result,x3_result, a1, b1, a2, b2, phi, saveFig=True, gif_fps=18, name=threeleg_dir+'threeleg.gif', iter=iter, len_iterarr = len_iter)

#import plot fns
from plot_results import plot_timeseries, plot_gait

#plot the time history of the states + controls
plot_timeseries(x1_result, x2_result, x3_result, x4_result, x5_result, x6_result,tau_result, time_result, dir=threeleg_dir)

# plot limit cycle
plot_gait(x1_result, x2_result, x3_result, x4_result, x5_result, x6_result, dir=threeleg_dir)



