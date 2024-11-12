import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

def control (model, delta_t):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
    'n_robust': 0,
    'n_horizon': 50,
    't_step': delta_t,
    'state_discretization': 'discrete',
    'store_full_solution':False,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }
    mpc.settings.supress_ipopt_output()
    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost'] # terminal cost
    lterm = model.aux['cost'] # terminal cost
    # stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(u=1e-2) # input penalty

    #max_x = np.array([[4.0], [10.0], [4.0], [10.0]])

    # lower bounds of the states
    #mpc.bounds['lower','_x','x'] = -max_x

    # upper bounds of the states
    #mpc.bounds['upper','_x','x'] = max_x

    # lower bounds of the input
    mpc.bounds['lower','_u','u'] = -50

    # upper bounds of the input
    mpc.bounds['upper','_u','u'] =  50

    mpc.setup()

    return mpc

