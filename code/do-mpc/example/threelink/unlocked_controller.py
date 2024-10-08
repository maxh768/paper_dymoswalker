import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc



def control_unlocked(model, delta_t=0.01):
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


    x1 = model.x['x1']
    x2 = model.x['x2']
    x3 = model.x['x3']
    x4 = model.x['dx1']
    x5 = model.x['dx2']
    x6 = model.x['dx3']
    # obj function
    mterm = (x1+0.106)**2 + (x2-0.326)**2 + (x3-0.331)**2
    lterm = (x1+0.106)**2 + (x2-0.326)**2 + (x3-0.331)**2
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # set r term ??
    mpc.set_rterm(
        tau_hip=1
    )

    # lower and upper bounds on states
    """mpc.bounds['lower','_x','x1'] = -1.5708 # -90 deg
    mpc.bounds['lower','_x','x2'] = -1.5708 # -90 deg
    mpc.bounds['upper','_x','x1'] = 1.5708 # +90 deg
    mpc.bounds['upper','_x','x2'] = 1.5708 # +90 deg
    mpc.bounds['upper','_x','x3'] = 1.5708 # +90 deg
    mpc.bounds['lower','_x','x3'] = -1.5708 # +90 deg"""

    # lower and upper bounds on inputs (tau/desired pos?)
    mpc.bounds['lower','_u','tau_hip'] = -3
    mpc.bounds['upper','_u','tau_hip'] = 3


    # should maybe add scaling to adjust for difference in magnitude from diferent states (optinal/future)

    # set uncertain parameters (none for now)

    mpc.setup()
    return mpc
