import numpy as np
from numpy.linalg import inv
from TS_solver import TSM

def force(w, x):
    # equation:
    # f = w_dot = A*w + B(w)
    #A = -Hinv*B
    #C = -Hinv*G
    #theta = [theta1; theta2]
    #w = [theta1; theta2; theta1_dot; theta2_dot]
    #w_dot = [theta1_dot; theta2_dot; theta1_dotdot; theta2_dotdot]

    # define x as [a,b]

    # set params
    a = x[0]
    b = x[1]
    mh = 10
    m = 5
    phi = 0.05
    l = a + b
    g = 9.81

    # set states
    w1 = w[0]
    w2 = w[1]
    w3 = w[2]
    w4 = w[3]

    A = np.zeros((4,4)) # empty forcing matrix

    # build forcing matrix
    H = np.zeros((2,2))
    H22 = (mh + m)*(l**2) + m*a**2
    H12 = -m*l*b*np.cos(w2 - w1)
    H21 = H12
    H11 = m*b**2
    H[0,0] = H11
    H[0,1] = H12
    H[1,0] = H21
    H[1,1] = H22
    #print(H)
    Hinv = inv(H)

    B = np.zeros((2,2))
    B12 = m*l*b*w4*np.sin(w2-w1)
    B21 = -m*l*b*w3*np.sin(w2-w1)
    B[0,1] = B12
    B[1,0] = B21


    A_im = np.dot(-Hinv,B)
    # finish forcing matrix
    A[0,2] = 1; A[1,3] = 1
    A[2,2] = A_im[0,0]
    A[2,3] = A_im[0,1]
    A[3,2] = A_im[1,0]
    A[3,3] = A_im[1,1]
    
    #print(A_im)
    #print(A)

    #force from gravity matrix (C):
    G = np.zeros(2)
    G[0] = m*g*b*np.sin(w1)
    G[1] = -(mh*l + m*(a+l))*np.sin(w2)
    C_int = np.dot(-Hinv,G)

    C = np.zeros(4)
    C[2] = C_int[0]
    C[3] = C_int[1]
    #print(C)

    f = np.zeros(4)
    f[:] += A.dot(w) + C[:]
    #print(f)
    
    # return total forcing term
    return f


# set up parameters for TSM
n = 5
Ndof = 4
x = [.5, .5]

# create TSM object
compass = TSM(force, n, Ndof, x=x)
xin = compass.generate_xin()
print(xin)
sol = compass.solve(xin)
print(sol)



    

    

