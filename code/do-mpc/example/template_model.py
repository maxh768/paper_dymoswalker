#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    a = 0.5
    b = 0.5
    mh = 10
    m = 5
    g = 9.81
    l = a + b
    phi = 0.05 # slope - 3 degrees

    # States struct (optimization variables):
    x1 = model.set_variable('_x',  'x1')  # back leg angle 
    x2 = model.set_variable('_x',  'x2')  # front leg angle
    x3 = model.set_variable('_x',  'x3')  # back leg velocity
    x4 = model.set_variable('_x',  'x4')  # front leg velocity
    
    # Input struct (optimization variables):
    tau = model.set_variable('_u',  'tau')

    # Fixed parameters:
    a = model.set_variable('_p',  'a')
    b = model.set_variable('_p', 'b')
    mh = model.set_variable('_p', 'mh')
    m = model.set_variable('_p', 'm')
    l = model.set_variable('_p', 'l')
    g = model.set_variable('_p', 'g')

    H22 = (mh + m)*(l**2) + m*a**2
    H12 = -m*l*b*np.cos(x2 - x1)
    H11 = m*b**2
    h = -m*l*b*np.sin(x1-x2)
    G2 = -(mh*l + m*a + m*l)*g*np.sin(x2)
    G1 = m*b*g*np.sin(x1)
    K = 1 / (H11*H22 - (H12**2)) # inverse constant

    # Differential equations
    model.set_rhs('x1', x3)
    model.set_rhs('x2', x4)
    model.set_rhs('x3', (H12*K*h*x3**2) + (H22*K*h*x4**2) - H22*K*G1 + H12*K*G2 - (H22 + H12)*K*tau)
    model.set_rhs('x4', (-H11*K*h*x3**2) - (H12*K*h*x4**2) + H12*K*G1 - H11*K*G2 + ((H12 + H11)*K*tau))

    # Build the model
    model.setup()

    return model
