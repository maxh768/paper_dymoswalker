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
    phi = 0.05,
    x=None
    ):
        self.force = force
        self.n = n
        self.Ndof = Ndof
        self.w = np.zeros(n*Ndof)
        self.T = T
        self.omega = 2*np.pi/T # dummy for now
        self.phi_constraint = -2*phi
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
        #print('----NEW RES----')
        w = self.w
        force = self.force
        n = self.n
        Ndof = self.Ndof
        phibound = np.zeros(2)
        

        """# find point where heel-strike occurs (compass gait specific)
        not working
        plan:
        1. run time spectral method with system without collision from initial point
        2. find where collision point is from converged solution (may need linear interpolation
        due to lack of small timestep)
        3. run time spectral method starting from collision states

        4. repeat for x # of cycles
        print(w)
        from calc_transition import calc_trans
        for j in range(n-1):
            w1_prev = w[j*Ndof]
            w2_prev = w[j*Ndof + 1]
            w3_prev = w[j*Ndof + 2]
            w4_prev = w[j*Ndof + 3]

            w1_cur = w[j*Ndof + 4]
            w2_cur = w[j*Ndof + 4 + 1]
            w3_cur = w[j*Ndof + 4 + 2]
            w4_cur = w[j*Ndof + 4 + 3]


            phibound[0] = w1_prev + w2_prev
            phibound[1] = w1_cur + w2_cur
            #print(phibound)
            #print('w1 prev: ',w1_prev_cur)
            #print('w1 current: ', w1_cur)
            if ((((phibound[0] > self.phi_constraint) and (phibound[1] < self.phi_constraint)) or ((phibound[0] < self.phi_constraint) and (phibound[1] > self.phi_constraint)))):
                print('----TRANSITION----')
                print([w1_cur, w2_cur, w3_cur, w4_cur])
                newstates = calc_trans(w1_cur,w2_cur,w3_cur,w4_cur)
                w[j*Ndof] = newstates[0]
                w[j*Ndof + 1] = newstates[1]
                w[j*Ndof + 2] = newstates[2]
                w[j*Ndof + 3] = newstates[3]
                break

        
        self.w[:] = w"""



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
            w0[i] = 0.3* np.sin(float(i) / float(n * Ndof) * 2.0 * np.pi)


        xin[4:] = w0[4:]
        #print(xin)

        return xin
    
    # solve Rdyn = 0 for state vars
    def solve(self, xin, tol=1e-10):

        #Rdyn = self.compute_Rdyn
        #print(Rdyn)
        sol = optimize.newton_krylov(self.res_wrapper, xin, f_tol=tol)
        #print(self.w)
        #print(sol)

        self.w[:] = sol[:]

        return sol






