import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

def model_set(M,m,L):
    g = 9.81
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
    dx = model.set_variable(var_type='_x', var_name='dx', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    xdd = (-m*g*np.sin(theta)*np.cos(theta) - (u + m*L*(dtheta**2)*np.sin(theta))) / (m*np.cos(theta)**2 - (M+m))
    thetadd = ((M+m)*g*np.sin(theta) + np.cos(theta)*(u + M*L*(dtheta**2)*np.sin(theta))) / (m*L*np.cos(theta)**2 - (M+m)*L)

    model.set_rhs('x', dx)
    model.set_rhs('theta', dtheta)
    model.set_rhs('dtheta', thetadd)
    model.set_rhs('dx', xdd)

    J = (x)**2 + 50*(theta - np.pi)**2 + (dx)**2 + (dtheta)**2
    model.set_expression(expr_name='cost', expr=J)
    

    # Build the model
    model.setup()

    return model