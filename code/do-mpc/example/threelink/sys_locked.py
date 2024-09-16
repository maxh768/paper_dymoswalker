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

    x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
    x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))

    dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
    dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))

    # control / inputs
    tau = model.set_variable(var_type='_u', var_name='tau', shape=(1,1))

    # set params
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

    # equations for locked knee dynamics
    H11 = ms*a1**2 + mt*(ls+a2)**2 + (mh + ms + mt)*L**2
    H12 = -(mt*b2 + ms*(lt+b1))*L*np.cos(x2-x1)
    H22 = mt*b2**2 + ms*(lt + b1)**2
    H21 = H12

    B12 = (-(mt*b2 + ms*(lt+b1))*L*np.sin(x1-x2))*dx2
    B21 = (mt*b2 + ms*(lt+b1))*L*np.sin(x1-x2)*dx1

    G1 = -(ms*a1 + mt*(ls+a2) + (mh+mt+ms)*L)*g*np.sin(x1)
    G2 = (mt*b2 + ms*(lt+b1))*g*np.sin(x2)

    K = 1 / ((H11*H22 - (H12**2)))

    H_I11 = H22/K
    H_I12 = -H12/K
    H_I21 = -H21/K
    H_I22 = H11/K

    # U matrix (torque)
    U1 = 0
    U2 = 0

    dx1set = -(H_I12*B21*dx1 + H_I11*B12*dx2) - (H_I11*G1 + H_I12*G2) #+ (H_I11*U1 + H_I12*U2)*tau
    dx2set = -(H_I22*B21*dx1 + H_I21*B12*dx2) - (H_I21*G1 + H_I22*G2) #+ (H_I12*U1 + H_I22*U2)*tau

    # set rhs
    model.set_rhs('x1',dx1)
    model.set_rhs('x2',dx2)
    model.set_rhs('dx1',dx1set)
    model.set_rhs('dx2',dx2set)

    model.setup()
    return model