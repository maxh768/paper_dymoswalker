import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
# Import do_mpc package:
import do_mpc

def model_locked():
    #set states
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)


    #set states
    x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
    x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))

    dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
    dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))

    # control / inputs
    tau = model.set_variable(var_type='_u', var_name='tau', shape=(1,1))

    # set params
    a = 0.5
    b = 0.5
    mh = 10
    m = 5
    phi = 0.05
    l = a + b
    g = 9.81

    # dynamics
    H22 = (mh + m)*(l**2) + m*a**2
    H12 = -m*l*b*np.cos(x2 - x1)
    H11 = m*b**2
    h = -m*l*b*np.sin(x1-x2)
    G2 = -(mh*l + m*a + m*l)*g*np.sin(x2)
    G1 = m*b*g*np.sin(x1)

    K = 1 / (H11*H22 - (H12**2)) # inverse constant
    dx1set = (H12*K*h*dx1**2) + (H22*K*h*dx2**2) - H22*K*G1 + H12*K*G2 - (H22 + H12)*K*tau
    dx2set = (-H11*K*h*dx1**2) - (H12*K*h*dx2**2) + H12*K*G1 - H11*K*G2 + ((H12 + H11)*K*tau)


    # set rhs
    model.set_rhs('x1',dx1)
    model.set_rhs('x2',dx2)
    model.set_rhs('dx1',dx1set)
    model.set_rhs('dx2',dx2set)

    model.setup()
    return model