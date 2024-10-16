import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

# set params
dt = .01
num_steps = 1000

# import model
from model import model
model, muj = model()
# import controller
from controller import controller
mpc = controller(model, dt=dt)

"""
## CONFIG SIMULATOR
"""
#unlocked sim
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)
# uncertain vars (future)
simulator.setup()


"""
CONTROL LOOP
"""


# IC
x0 = -6
theta0 = 0.2

# set mujoco model to start at IC
state = np.array([x0, theta0, 0, 0])
muj.set_state(state=state)
checkstate = muj.get_state()
#print(checkstate) #need a way to check the state

# set mpc controller initial guess from IC
x0 = np.array([x0, theta0]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

"""
attempt to simulate
"""
#muj.render()
for i in range(num_steps):
    obs = muj.get_state()
    #print('obs:', obs)
    x0[0] = obs[0]
    x0[1] = obs[1]

    u0 = mpc.make_step(x0)
    action = np.zeros(1)
    print(u0)
    #print(x0)
    action[0] = u0
    muj.step(action)
    x0 = simulator.make_step(u0)
    #print(x0)

muj.close()
