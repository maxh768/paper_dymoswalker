import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def controller(model, dt=0.1):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 70,
        't_step': dt,
        'n_robust': 1,
        'store_full_solution': True,
        #'supress_ipopt_output': True
    }
    mpc.settings.supress_ipopt_output()
    mpc.set_param(**setup_mpc)


    x = model.x['x']
    theta = model.x['theta']

    # obj function
    mterm = (theta)**2
    lterm = (theta)**2
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # set r term ??
    mpc.set_rterm(
        F=1
    )

    # lower and upper bounds on states

    # lower and upper bounds on inputs (tau/desired pos?)
    mpc.bounds['lower','_u','F'] = -3
    mpc.bounds['upper','_u','F'] = 3



    # should maybe add scaling to adjust for difference in magnitude from diferent states (optinal/future)

    # set uncertain parameters (none for now)

    mpc.setup()
    return mpc