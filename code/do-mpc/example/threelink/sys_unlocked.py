import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
# Import do_mpc package:
import do_mpc


def model_unlocked():
    #set states
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
    x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))
    x3 = model.set_variable(var_type='_x', var_name='x3',shape=(1,1))

    dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
    dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))
    dx3 = model.set_variable(var_type='_x', var_name='dx3', shape=(1,1))

    # control / inputs
    tau_hip = model.set_variable(var_type='_u', var_name='tau_hip', shape=(1,1)) # torque at hip
    #tau_knee = model.set_variable(var_type='_u', var_name='tau_knee', shape=(1,1)) # torque at knee
    #tau_ankle = model.set_variable(var_type='_u', var_name='tau_ankle', shape=(1,1)) # torque at ankle



    #print('x1={}, with x1.shape={}'.format(x1, x1.shape))
    #print('x2={}, with x2.shape={}'.format(x2, x1.shape))
    #print('dx1={}, with dx1.shape={}'.format(dx1, dx1.shape))
    #print('dx2={}, with dx2.shape={}'.format(dx2, dx2.shape))

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
    phi = .05


    # dynamic matrix (will be inverted)
    H11 = ms*a1**2 + mt*(ls+a2)**2 + (mh+ms+mt)*L**2
    H12 = -(mt*b2 + ms*lt)*L*np.cos(x2-x1)
    H13 = -ms*b1*L*np.cos(x3-x1)
    H22 = mt*b2**2 + ms*lt**2
    H23 = ms*lt*b1*np.cos(x3-x2)
    H33 = ms*b1**2
    H21 = H12
    H31 = H13
    H32 = H23

    # B matrix
    B11 = 0
    B12 = (-(mt*b2+ms*lt)*L*np.sin(x1-x2))*dx2
    B13 = (-ms*b1*L*np.sin(x1-x3))*dx3
    B21 = ((mt*b2+ms*lt)*L*np.sin(x1-x2))*dx1
    B22 = 0
    B23 = (ms*lt*b1*np.sin(x3-x2))*dx3
    B31 = (ms*b1*L*np.sin(x1-x3))*dx1
    B32 = (-ms*lt*b1*np.sin(x3-x2))*dx2
    B33 = 0

    # Gravity Matrix
    G1 = -(ms*a1+mt*(ls+a2)+(mh+ms+mt)*L)*g*np.sin(x1)
    G2 = (mt*b2 + ms*lt)*g*np.sin(x2)
    G3 = (ms*b1*g*np.sin(x3))

    # inverting H matrix
    #determinate
    detH = H11*(H33*H22 - H23**2) - H12*(H33*H12 - H23*H13) + H13*(H23*H12 - H22*H13)
    # inverse terms
    H_I11 = (H33*H22 - H23**2)/detH
    H_I12 = (H13*H23 - H33*H12)/detH
    H_I13 = (H12*H23 - H13*H22)/detH
    H_I22 = (H33*H11 - H13**2)/detH
    H_I23 = (H12*H13 - H11*H23)/detH
    H_I33 = (H11*H22 - H12**2)/detH
    H_I21 = H_I12
    H_I31 = H_I13
    H_I32 = H_I23



    dx1set = -( (H_I12*B21 + H_I13*B31)*dx1 + (H_I11*B12 + H_I13*B32)*dx2 + (H_I11*B13 + H_I12*B23)*dx3 ) - (H_I11*G1 + H_I12*G2 + H_I13*G3) + (H_I11-H_I12)*-tau_hip
    dx2set = -( (H_I22*B21 + H_I23*B31)*dx1 + (H_I21*B12 + H_I23*B32)*dx2 + (H_I21*B13 + H_I22*B23)*dx3 ) - (H_I21*G1 + H_I22*G2 + H_I23*G3) + (H_I21-H_I22)*-tau_hip
    dx3set = -( (H_I32*B21 + H_I33*B31)*dx1 + (H_I31*B12 + H_I33*B32)*dx2 + (H_I31*B13 + H_I32*B23)*dx3 ) - (H_I31*G1 + H_I32*G2 + H_I33*G3) #+ (H_I31-H_I32)*tau_hip

    # set rhs
    model.set_rhs('x1',dx1)
    model.set_rhs('x2',dx2)
    model.set_rhs('x3',dx3)
    model.set_rhs('dx1',dx1set)
    model.set_rhs('dx2',dx2set)
    model.set_rhs('dx3',dx3set)


    model.setup()
    return model