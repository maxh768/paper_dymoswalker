import numpy as np
import openmdao.api as om
import dymos as dm
from leggeddynamics import lockedKneeDynamics


def main():
    L = 1
    a1 = 0.375
    a2 = 0.175
    b1 = 0.125
    b2 = 0.325
    m_H = 0.5
    m_t = 0.5
    m_s = 0.05