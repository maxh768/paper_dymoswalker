import numpy as np
from scipy import optimize

class TSM(object):

    # init
    def __init__(
    self,
    force,
    n,
    Ndof,
    T,
    x=None
    ):
        self.force = force
        self.n = n
        self.Ndof = Ndof
        self.w = np.zeros(n*Ndof)
        self.T = T
        self.omega = 2*np.pi/T # dummy for now
        if x is None:
            x = [0.5, 0.5]
        else:
            self.x = x



    # wrapper function to import to scipy
    def res_wrapper(self, u):
        self.set_state(u)
        res = self.compute_Rdyn()
        return res

    # sets state w = u
    def set_state(self, u):
        self.w[:] = u


    # set up residual
    def compute_Rdyn(self):
        w = self.w
        force = self.force
        n = self.n
        Ndof = self.Ndof

        wdot = self.compute_dwdt()

        Rdyn = np.zeros(n*Ndof)

        for i in range(n):

            w_loc = w[i * Ndof : (i + 1) * Ndof]
            Rdyn[i * Ndof : (i + 1) * Ndof] = -force(w_loc, self.x)[:]

        Rdyn[:] += wdot[:]
        #print(Rdyn)
        return Rdyn

    # compute wdotn = D*wn
    def compute_dwdt(self):
        omega = self.omega
        n = self.n
        Ndof = self.Ndof
        D = self.compute_Dmatrix()

        wdot = np.zeros(n*Ndof)

        for i in range(Ndof):
            wdof = np.zeros(n)
            for j in range(n):
                wdof[j] = self.w[j * Ndof + i]
                #print(wdof[j])

            wdof_der = D.dot(wdof)

            for j in range(n):
                wdot[j * Ndof + i] = wdof_der[j]

        return wdot


    # compute time derivitve matrix D (nxn)
    def compute_Dmatrix(self):
        omega = self.omega
        n = self.n
        D = np.zeros((n,n))
        if (n%2==0):
            # even
            for i in range(n):
                for j in range(n):
                    if (i != j):
                        D[i, j] = 0.5 * (-1)**(i-j) / np.tan(np.pi*(i - j) / n)

        else:
            # odd
            for i in range(n):
                for j in range(n):
                    if (i != j):
                        D[i, j] = 0.5 * (-1)**(i - j) / np.sin(np.pi*(i - j) / n)

        D = D * omega

        return D
    

    # make a initial guess for the states (not sure how yet)
    def generate_xin(self):
        n = self.n
        Ndof = self.Ndof


        w0 = np.zeros(n * Ndof)
        xin = np.zeros(n * Ndof )

        states_init = [-0.3, 0.2038, -0.41215, -1.0501]
        xin[0:4] = states_init[:]
        
        for i in range(n* Ndof):
            w0[i] = 4* np.sin(float(i) / float(n * Ndof) * 2.0 * np.pi)


        xin[4:] = w0[4:]
        #print(xin)

        return xin
    
    # solve Rdyn = 0 for state vars
    def solve(self, xin, tol=1e-10):

        #Rdyn = self.compute_Rdyn
        #print(Rdyn)
        sol = optimize.newton_krylov(self.res_wrapper, xin, f_tol=tol)

        self.w[:] = sol[:]

        return sol






