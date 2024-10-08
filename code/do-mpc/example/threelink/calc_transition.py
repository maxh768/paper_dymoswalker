import numpy as np
from numpy.linalg import inv

def calc_trans(x1, x2, x3, x4, mh=.5, m1=.05, m2=.5, a1=.375, a2=.175, b1=0.125, b2=0.325):
    x1 = x1[0]
    x2 = x2[0]
    x3 = x3[0]
    x4 = x4[0]
    
    l1 = a1+b1
    l2 = a2+b2
    L = l1+l2

    alpha = np.abs((x2 - x1)) # angle between legs
    alpha = float(alpha)

    # plus matrix
    Qp11 = -(m1*(b1+l2) + m2*b2)*L*np.cos(alpha) + m2*(l1+a2)**2 + (mh+m1+m2)*L**2 + m1*a1**2
    Qp12 = -(m1*(b1+l2) + m2*b2)*L*np.cos(alpha) + m1*(l2+b1)**2 + m2*b2**2
    Qp21 = -(m1*(b1+l2) + m2*b2)*L*np.cos(alpha)
    Qp22 = m1*(l2+b1)**2 + m2*b2**2

    # minus matrix
    Qm12 = -m1*a1*(l2+b1) - m2*b2*(l1+a2)
    Qm11 = (mh*L + 2*m2*(a2+l1) + m1*a1)*L*np.cos(alpha) + Qm12
    Qm21 = Qm12
    Qm22 = 0


    Qplus = np.array([[Qp11, Qp12], [Qp21, Qp22]])
    Qminus = np.array([[Qm11, Qm12], [Qm21, Qm22]])

    #print(Qplus, Qminus)
    Qplus_inverted = inv(Qplus)

    # H transition matrix
    H = np.dot(Qplus_inverted, Qminus)

    # new x3 and x4
    newx3 = H[0,0]*x3 + H[0, 1]*x4
    newx4 = H[1,0]*x3 + H[1, 1]*x4

    new_states_init = [x2, x1, newx3, newx4] # check this
    return new_states_init