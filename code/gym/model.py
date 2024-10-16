import numpy as np
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
# Import do_mpc package:
import do_mpc
from casadi import *

from gymenv import Pendulum


def model():
    # set up gym env
    muj = Pendulum()
    muj.reset()


    # set up dompc model

    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    #set states
    x = model.set_variable(var_type='_x', var_name='x', shape=(1,1)) # x pos of cart
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1)) # vertical angle of pole


    # set input force
    F = model.set_variable(var_type='_u', var_name='F', shape=(1,1))


    obs = muj.get_state()
    xnext = obs[2]*-F
    cas_xnext = SX.ones((1,1))
    cas_xnext[0] = xnext
    thetanext = obs[3]*-F
    cas_thetanext = SX.ones((1,1))
    cas_thetanext[0] = thetanext


    # set rhs of states
    model.set_rhs('x',cas_xnext)
    model.set_rhs('theta',cas_thetanext)
    model.setup()

    return model, muj




