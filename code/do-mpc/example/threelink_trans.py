import numpy as np
from numpy.linalg import inv

def kneestrike(x1, x2, x3, x4, x5, x6,a1=0.375, a2=0.175, b1=0.125, b2=0.325, mh=0.5, mt=0.5, mh=0.5):
    
    ls = a1+b1
    lt = a2+b2
    L = ls + lt
    
    alpha = np.cos(x1-x2)
    beta = np.cos(x1-x3)
    gamma = np.cos(x2-x3)

    # pre matrix
    Q11_minus = -(ms*lt + mt*b2)*L*np.cos(alpha) - ms*b1*L*np.cos(beta) + (mt + ms +mh)*L**2 + ms*a1**2 + mt*(ls+a2)**2
    Q12_minus = -(ms*lt + mt*b2)*L*np.cos(alpha) + ms*b1*lt*np.cos(gamma) + mt*b2**2 + ms*lt**2
    Q13_minus = -ms*b1*L*np.cos(beta) + ms*b1*lt*np.cos(gamma) + mt*b2**2 + ms*lt**2
    Q21_minus = -(ms*lt + mt*b2)*L*np.cos(alpha) - ms*b1*L*np.cos(beta)
    Q22_minus = ms*b1*lt*np.cos(gamma) + ms*lt**2 + mt*b2**2
    Q23_minus = ms*b1*lt*np.cos(gamma) + ms*b1**2

    # post matrix
    Q22_plus = ms*(lt+b1)**2 + mt*b2**2
    Q21_plus = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha)
    Q12_plus = Q21_plus + ms*(lt+b1)**2 + mt*b2**2
    Q11_plus = Q21_plus + mt*(ls+a2)**2 + (mh + mt + ms)*L**2 + ms*a1**2

    Q_minus = np.array([[Q11_minus, Q12_minus, Q13_minus], [Q21_minus, Q22_minus, Q23_minus]])
    Q_plus = np.array([[Q11_plus, Q22_]])