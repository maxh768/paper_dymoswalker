import numpy as np
from numpy.linalg import inv

def calc_trans(x1, x2, x3, x4, m=5, mh=10, a=0.5, b=0.5):
    x1 = x1[0]
    x2 = x2[0]
    x3 = x3[0]
    x4 = x4[0]
    
    l = a+b

    alpha = np.abs((x2 - x1)) / 2 # angle between legs
    alpha = float(alpha)
    # Q+ matrix
    Qp11 = m*b*(b-l*np.cos(2*alpha))
    Qp12 = m*l*(l-b*np.cos(2*alpha)) + m*a**2 + mh*l**2
    Qp21 = m*b**2
    Qp22 = -m*b*l*np.cos(2*alpha)
  
    #Q- matrix
    Qm11 = -m*a*b
    Qm12 = -m*a*b + (mh*l**2 + 2*m*a*l)*np.cos(2*alpha)
    Qm21 = 0
    Qm22 = -m*a*b
        
    Qplus = np.array([[Qp11, Qp12], [Qp21, Qp22]])
    Qminus = np.array([[Qm11, Qm12], [Qm21, Qm22]])

    #print(Qplus, Qminus)
    Qplus_inverted = inv(Qplus)

    # H transition matrix
    H = np.dot(Qplus_inverted, Qminus)

    # new x3 and x4
    newx3 = H[0,0]*x3 + H[0, 1]*x4
    newx4 = H[1,0]*x3 + H[1, 1]*x4

    new_states_init = [x2, x1, newx3, newx4]
    return new_states_init