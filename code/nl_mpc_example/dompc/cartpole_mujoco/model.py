import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

def model_set(A, B):
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    xdot = A @ x + B @ u
    

    model.set_rhs('x', xdot)

    # cost (need to change)
    J = 0.2*(x[0])**2 + (x[1])**2 + 0.1*(x[2])**2 + 0.1*(x[3])**2 
    model.set_expression(expr_name='cost', expr=J)
    

    # Build the model
    model.setup()

    return model