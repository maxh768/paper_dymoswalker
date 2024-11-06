import numpy as np

def discretize_sys(theta, dtheta, h, m1=10, m2=2, L=.5, g=9.81,):
    A = np.matrix([ [1, 0, h, 0], [0, 1, 0, h], [0, 0, 1, 0], [0, 0, 0, 1] ])
    K = 1 / (m2*L*(np.cos(theta)**2) - L*(m1+m2))

    B = (K*h)*np.matrix([ [-h*L], [h*np.cos(theta)], [-L], [np.cos(theta)]   ])

    c3 = -g*m2*L*np.cos(theta)*np.sin(theta)-(L**2)*m2*(dtheta**2)*np.sin(theta)
    c4 = (m1+m2)*g*np.sin(theta) + m2*L*(dtheta**2)*np.cos(theta)*np.sin(theta)
    c1 = h*c3
    c2 = h*c4

    C = (h*K)*np.matrix([[c1],[c2],[c3],[c4]])
    return A, B, C

if __name__ == '__main__':
    x = 0.2
    theta = np.deg2rad(30)
    dtheta = np.deg2rad(-5)
    h = 0.0005
    A, B, C = discretize_sys(theta, dtheta, h)
    print('A: ',A)
    print('B: ',B)
    print('C: ',C)