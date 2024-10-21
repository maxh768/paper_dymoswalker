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
    


    #force from gravity matrix (C):
    G = np.zeros(2)
    G[0] = m*g*b*np.sin(w1)
    G[1] = -(mh*l + m*(a+l))*np.sin(w2)
    C_int = np.dot(-Hinv,G)

    C = np.zeros(4)
    C[2] = C_int[0]
    C[3] = C_int[1]


    f = np.zeros(4)
    f[:] += A.dot(w) + C[:]
    
    # return total forcing term
    return f


# set up parameters for TSM
n = 30
Ndof = 4
x = [.5, .5]
T = 0.25

phi = 0.05 # set ramp angle
phi_constraint = -2*phi

# create TSM object for pre collision
compass_precol = TSM(force, n, Ndof,T, x=x,phi=phi)
xin = compass_precol.generate_xin()
sol = compass_precol.solve(xin)
print(sol)

# find collision point
phibound = np.zeros(2)
from calc_transition import calc_trans
for j in range(n-1):
    w1_prev = sol[j*Ndof]
    w2_prev = sol[j*Ndof + 1]
    w3_prev = sol[j*Ndof + 2]
    w4_prev = sol[j*Ndof + 3]

    w1_cur = sol[j*Ndof + 4]
    w2_cur = sol[j*Ndof + 4 + 1]
    w3_cur = sol[j*Ndof + 4 + 2]
    w4_cur = sol[j*Ndof + 4 + 3]


    phibound[0] = w1_prev + w2_prev
    phibound[1] = w1_cur + w2_cur
    #print(phibound)
    #print('w1 prev: ',w1_prev_cur)
    #print('w1 current: ', w1_cur)
    if ((((phibound[0] > phi_constraint) and (phibound[1] < phi_constraint)) or ((phibound[0] < phi_constraint) and (phibound[1] > phi_constraint)))) and (j>2):
        print('----TRANSITION----')
        print([w1_cur, w2_cur, w3_cur, w4_cur])
        newstates = calc_trans(w1_cur,w2_cur,w3_cur,w4_cur)
        print(newstates)
"""
# create object for 2nd run of TSM
compass_postcol = TSM(force, n, Ndof,T, x=x,phi=phi)
xin[0:4] = newstates[:]
#print(xin)
sol = compass_postcol.solve(xin)
#print(sol)
"""




# data management and plot
x1arr = np.zeros(n)
x2arr = np.zeros(n)
x3arr = np.zeros(n)
x4arr = np.zeros(n)

for i in range(n):
    cx1 = sol[i*Ndof]
    x1arr[i] = cx1

    cx2 = sol[i*Ndof+ 1]
    x2arr[i] = cx2

    cx3 = sol[i*Ndof + 2]
    x3arr[i] = cx3

    cx4 = sol[i*Ndof + 3]
    x4arr[i] = cx4

import matplotlib.pyplot as plt
import matplotlib as matplotlib
#import niceplots

#plt.style.use(niceplots.get_style())

my_blue = '#4C72B0'
my_red = '#C54E52'
my_green = '#56A968'
my_brown = '#b4943e'

#upper link of one leg
fig, ax = plt.subplots()
ax.plot(x1arr, x3arr, linewidth=1.0, label='swing')
ax.plot(x2arr, x4arr, linewidth=1.0, label='stance')
ax.set_title('Limit Cycle With Time Spectral Method')
ax.set_xlabel('angle')
ax.set_ylabel('angular velocity')
ax.legend()
dir = './research_template/time_spectral_images/'
plt.savefig(dir+'TS')




    

    

