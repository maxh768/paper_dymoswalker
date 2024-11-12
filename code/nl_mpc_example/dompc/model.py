import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

def model_set(X0, h):
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    from cartpole_sys import discretize_sys
    theta = float(X0[1])
    dtheta = float(X0[3])
    A, B, C = discretize_sys(theta, dtheta, h)


    x_next = A@_x + B@_u + C
    #print(A)
    #print(B)
    #print(C)

    tar_theta = np.deg2rad(180)
    model.set_rhs('x', x_next)

    J = (_x[0]**2) + (_x[1] - tar_theta)**2 + (_x[2]**2) + (_x[3]**2)
    model.set_expression(expr_name='cost', expr=J)
    

    # Build the model
    model.setup()

    return model