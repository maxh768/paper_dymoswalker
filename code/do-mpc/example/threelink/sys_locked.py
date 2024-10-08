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
    a1 = 0.375
    b1 = 0.125
    a2 = 0.175
    b2 = 0.325
    mh = 0.5
    m1 = 0.05
    m2 = 0.5
    g = 9.81
    l1 = a1+b1
    l2 = a2+b2
    L = a1+b1+a2+b2
    phi = 0.05

    # H base matrix
    H11 = m1*a1**2 + m2*(l1+a2)**2 +(mh+m1+m2)*L**2
    H12 = -(m2*b2*L + m1*L*(l2+b1))*np.cos(x2-x1)
    H21 = H12
    H22 = m2*b2**2 + m1*(l2+b1)**2

    # B base matrix
    B11 = 0
    B22 = 0
    B12 = (-m2*b2*L - m1*L*(l2+b1))*dx2*np.sin(x1-x2)
    B21 = (m2*b2*L + m1*L*(l2+b1))*dx1*np.sin(x1-x2)

    # G base matrix
    g1 = -(mh+m1+m2)*g*L*np.sin(x1) - m1*g*a1*np.sin(x1) - m2*g*(l1+a2)*np.sin(x1)
    g2 = (m2*b2 + m1*(l2+b1))*g*np.sin(x2)

    # inverse of H matrix
    k = 1 / (H11*H22 - H12*H21)
    HI_11 = H22*k
    HI_12 = -H12*k
    HI_21 = -H21*k
    HI_22 = H11*k

    dx1set = -(HI_12*B21*dx1 + HI_11*B12*dx2) - (HI_11*g1 + HI_12*g2) # add forcing term
    dx2set = -(HI_22*B21*dx1 + HI_21*B12*dx2) - (HI_21*g1 + HI_22*g2) # add forcing term



    # set rhs
    model.set_rhs('x1',dx1)
    model.set_rhs('x2',dx2)
    model.set_rhs('dx1',dx1set)
    model.set_rhs('dx2',dx2set)

    model.setup()
    return model