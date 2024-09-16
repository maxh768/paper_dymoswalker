import numpy as np
from numpy.linalg import inv

def kneestrike(x1, x2, x3, x4, x5, x6,a1=0.375, a2=0.175, b1=0.125, b2=0.325, mh=0.5, mt=0.5, ms=0.05):
    x1 = x1[0]
    x2 = x2[0]
    x3 = x3[0]
    x4 = x4[0]
    x5 = x5[0]
    x6 = x6[0]

    ls = a1+b1
    lt = a2+b2
    L = ls + lt
    
    alpha = np.cos(x1-x2)
    beta = np.cos(x1-x3)
    gamma = np.cos(x2-x3)

    # pre matrix
    Q11_minus = -(ms*lt + mt*b2)*L*np.cos(alpha) - ms*b1*L*np.cos(beta) + (mt + ms +mh)*L**2 + ms*a1**2 + mt*(ls+a2)**2
    Q12_minus = -(ms*lt + mt*b2)*L*np.cos(alpha) + ms*b1*lt*np.cos(gamma) + mt*b2**2 + ms*lt**2
    Q13_minus = -ms*b1*L*np.cos(beta) + ms*b1*lt*np.cos(gamma) + ms*b1**2
    Q21_minus = -(ms*lt + mt*b2)*L*np.cos(alpha) - ms*b1*L*np.cos(beta)
    Q22_minus = ms*b1*lt*np.cos(gamma) + ms*lt**2 + mt*b2**2
    Q23_minus = ms*b1*lt*np.cos(gamma) + ms*b1**2

    # post matrix
    Q21_plus = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha)
    Q12_plus = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha) + (ms*(lt+b1)**2 + mt*b2**2)
    Q11_plus = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha) + (mt*(ls+a2)**2 + (mh + mt + ms)*L**2 + ms*a1**2)
    Q22_plus = ms*(lt+b1)**2 + mt*(b2**2)

    Q_minus = np.array([[Q11_minus, Q12_minus, Q13_minus], [Q21_minus, Q22_minus, Q23_minus]])
    Q_plus = np.array([[Q11_plus, Q12_plus], [Q21_plus, Q22_plus]])
    
    Q_plus_inv = inv(Q_plus)

    H = np.dot(Q_plus_inv, Q_minus)
    #print(H)
    oldvelo = np.array([[x4], [x5], [x6]])
    #print('oldvelo: ', oldvelo)

    newstates = np.dot(H, oldvelo)
    newx1 = x1
    newx2 = x2
    newx3 = newstates[0,0]
    newx4 = newstates[1,0]


    new_states_init = [newx1, newx2, newx3, newx4]
    #print(new_states_init)
    return new_states_init

def heelstrike(x1, x2, x3, x4,a1=0.375, a2=0.175, b1=0.125, b2=0.325, mh=0.5, mt=0.5, ms=0.05):
    x1 = x1[0]
    x2 = x2[0]
    x3 = x3[0]
    x4 = x4[0]

    ls = a1+b1
    lt = a2+b2
    L = ls + lt
    alpha = np.cos(x1 - x2)

    Q11m = -ms*a1*(lt+b1) + mt*b2*(ls + a2) + (mh*L + 2*mt*(a2+ls) + ms*a1)*L*np.cos(alpha)
    Q12m = -ms*a1*(lt+b1) + mt*b2*(ls + a2)
    Q21m = Q12m
    Q22m = 0

    Q11p = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha) + (ms+mt+mh)*L**2 + ms*a1**2 + mt*(a2+ls)**2
    Q12p = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha) + ms*(b1+lt)**2 + mt*(b2**2)
    Q21p = -(ms*(b1+lt) + mt*b2)*L*np.cos(alpha)
    Q22p = ms*(lt+b1)**2 + mt*b2**2

    Qp = np.array([[Q11p, Q12p], [Q21p, Q22p]])
    Qm = np.array([[Q11m, Q12m], [Q21m, Q22m]])

    Qp_inv = inv(Qp)

    H = np.dot(Qp_inv, Qm)

    oldvelo = np.array([[x3], [x4]])

    newstates = np.dot(H, oldvelo)

    newx1 = x2
    newx2 = x1
    newx3 = x1
    newx4 = newstates[0,0]
    newx5 = newstates[1,0]
    newx6 = newstates[1,0]
    new_states_init = [newx1, newx2, newx3, newx4, newx5, newx6]
    return new_states_init