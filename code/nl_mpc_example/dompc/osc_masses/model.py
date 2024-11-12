import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

def model_set():
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    A = np.array([[ 0.763,  0.460,  0.115,  0.020],
                [-0.899,  0.763,  0.420,  0.115],
                [ 0.115,  0.020,  0.763,  0.460],
                [ 0.420,  0.115, -0.899,  0.763]])

    B = np.array([[0.014],
                [0.063],
                [0.221],
                [0.367]])

    x_next = A@_x + B@_u

    model.set_rhs('x', x_next)

    model.set_expression(expr_name='cost', expr=sum1(_x**2))

    # Build the model
    model.setup()

    return model