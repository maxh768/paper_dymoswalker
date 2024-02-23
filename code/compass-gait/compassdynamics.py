import numpy as np
import openmdao.api as om
from numpy.linalg import inv
from numpy.linalg import multi_dot

np.seterr(divide='ignore', invalid='ignore')

class system(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)

    def setup(self):
        nn=self.options['num_nodes']

        input_names = ['a', 'b', 'x1', 'x2', 'x3', 'x4', 'mh', 'm']
        self.add_subsystem('lockedknee', dynamics(num_nodes=nn, ), promotes_inputs=input_names, promotes_outputs=['*'])
        self.add_subsystem('cost', CostFunc(num_nodes=nn, states_ref=self.options['states_ref'] ), promotes_inputs=['x1', 'x2', 'tau'], promotes_outputs=['*'])



class dynamics(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)
        self.options.declare('g', default=9.81, desc='gravity constant')

    def setup(self):
        nn = self.options['num_nodes']

        # inputs

        # length paramters
        self.add_input('a',shape=(1,),units='m')
        self.add_input('b', shape=(1,),units='m')


        """
        x1 = q1, x2 = q2, x3 = q1_dot, x4 = q2_dot
        """
        self.add_input('x1', shape=(nn,),units='rad', desc='q1')
        self.add_input('x2', shape=(nn,),units='rad', desc='q2')
        self.add_input('x3', shape=(nn,),units='rad/s', desc='q1 dot')
        self.add_input('x4', shape=(nn,),units='rad/s', desc='q2 dot')

        

        # masses
        self.add_input('m', shape=(1,),units='kg')
        self.add_input('mh', shape=(1,),units='kg')


        # applied torque - torque is applied equally and opposite to each leg
        #self.add_input('tau', shape=(nn,),units='N*m', desc='applied toruqe at hip')


        # outputs
        """
        from state space eqns
        """
        self.add_output('x1_dot', shape=(nn,), units='rad/s',desc='q1 dot')
        self.add_output('x2_dot', shape=(nn,), units='rad/s',desc='q2 dot')
        self.add_output('x3_dot', shape=(nn,), units='rad/s**2',desc='q1 dotdot')
        self.add_output('x4_dot', shape=(nn,), units='rad/s**2',desc='q2 dotdot')

        # transition eqs
        #self.add_output('alpha', shape=(nn,), units='rad', desc='angle between legs')
        self.add_output('phi_bounds', shape=(nn,), units='rad', desc='phi contraint equation')
        self.add_output('alpha_bounds', shape=(nn,), units='rad', desc='alpha contraint equation')
        self.add_output('x3changer', shape=(nn,), units='rad', desc='multiplier for x3 to change at transition')
        self.add_output('x4changer', shape=(nn,), units='rad', desc='multiplier for x4 to change at transition')
        
        
        #partials
        self.declare_partials(of=['*'], wrt=['x1', 'x2', 'x3', 'x4'], method='cs')#, rows=np.arange(nn), cols=np.arange(nn))
        self.declare_coloring(wrt=['x1', 'x2', 'x3', 'x4'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['x1', 'x2', 'x3', 'x4'], method='fd', step=1e-6)

        self.declare_partials(of=['*'], wrt=['m', 'mh', 'a', 'b'], method='cs')# rows=np.arange(nn), cols=np.arange(nn))
        self.declare_coloring(wrt=['m', 'mh', 'a', 'b'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['m', 'mh', 'a', 'b'], method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        g = self.options['g']
        a = inputs['a']
        b = inputs['b']
        mh = inputs['mh']
        m = inputs['m']
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        #tau = inputs['tau']

        l = a + b

        # q1 = Ons = x1, q2 = Os = x2 --- x1 = q1 is the back leg, x2 = q2 is the front leg, angles measured +CCW from vertical axis
        # mtrix components
        H22 = (mh + m)*(l**2) + m*a**2
        H12 = -m*l*b*np.cos(x2 - x1)
        H11 = m*b**2
        h = -m*l*b*np.sin(x1-x2)
        G2 = -(mh*l + m*a + m*l)*g*np.sin(x2)
        G1 = m*b*g*np.sin(x1)

        K = 1 / (H11*H22 - (H12**2)) # inverse constant

        outputs['x1_dot'] = x3
        outputs['x2_dot'] = x4
        outputs['x3_dot'] = (H12*K*h*x3**2) + (H22*K*h*x4**2) - H22*K*G1 + H12*K*G2 #- (H22 + H12)*K*tau
        outputs['x4_dot'] = (-H11*K*h*x3**2) - (H12*K*h*x4**2) + H12*K*G1 - H11*K*G2 #+ ((H12 + H11)*K*tau)

        # calculating cooridnates of points relative to stance foot
        # this will allow us to find alpha at any point in time
        xhip = -l*np.sin(x2) #changes sign
        yhip = l*np.cos(x2) # always positive
        x_hip2swing = l*np.sin(x1) # changes sign
        y_hip2swing = -l*np.cos(x1) # always negative
        x_swing = xhip+x_hip2swing # x and y coords of swing foot wrt stance foot
        y_swing = yhip+y_hip2swing # always positive
        L_stance2swing = np.sqrt(x_swing**2+y_swing**2)
        theta_inter = (np.pi/2) - np.tan(y_swing/np.abs(x_swing))
        theta_R = theta_inter - x2 # the angle between the two legs on the right side of the triangle

        alpha = (np.arcsin(L_stance2swing*np.sin(theta_R) / l)) / 2 # alpha - half the angle between the legs at hip
        
        # auxillary outputs for transition and bounds
        outputs['phi_bounds'] = x1 + x2
        outputs['alpha_bounds'] = -2*alpha + x2 - x1 #might be -2alpha instead of 2 alpha - TEST

        # calculating transition matrices for phase change
        Q11_m = -m*a*b; Q12_m = -m*a*b + ((mh*l**2) + 2*m*a*l)*np.cos(2*alpha); Q22_m = Q11_m
        Q11_p = m*b*(b - l*np.cos(2*alpha)); Q12_p = m*l*(l-b*np.cos(2*alpha)) + m*a**2 + mh*l**2; Q21_p = m*b**2; Q22_p = -m*b*l*np.cos(2*alpha)

        kq_p = 1 / (Q11_p*Q22_p - Q12_p*Q21_p) # inverse constant

        # transition matrix for velocities
        P11 = kq_p*(Q22_p*Q11_m ); P12 = kq_p*(Q22_p*Q12_m - Q12_p*Q22_m); P21 = kq_p*(-Q21_p*Q11_m); P22 = kq_p*(Q11_p*Q22_m)
        
        outputs['x3changer'] = P11*x3 + P12*x4
        outputs['x4changer'] = P21*x3 + P22*x4

    """def compute_partials(self, inputs, partials):
       # computes analytical partials 

        g = self.options['g']
        a = inputs['a']
        b = inputs['b']
        mh = inputs['mh']
        m = inputs['m']
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        #tau = inputs['tau']

        l = a + b

        # q1 = Ost = x1, q2 = Osw = x2
        # mtrix components
        H11 = (mh + m)*(l**2) + m*a**2
        H12 = -m*l*b*np.cos(x1 - x2)
        H22 = m*b**2
        h = m*l*b*np.sin(x1-x2)
        G1 = (mh*l + m*a + m*l)*g*np.sin(x1)
        G2 = -m*b*g*np.sin(x2)

        K = 1 / (H11*H22 - H12**2) # inverse constant

        # partials of terms wrt q1 q2
        H12_dq1 = m*l*b*np.sin(x1-x2)
        H12_dq2 = -m*l*b*np.sin(x1-x2)
        h_dq1 = m*l*b*np.cos(x1-x2)
        h_dq2 = -m*l*b*np.cos(x1-x2)
        G1_dq1 = (mh*l + m*a + m*l)*g*np.cos(x1)
        G2_dq2 = -m*b*g*np.cos(x2)

        #H12_abs = -H12

        K_dq1 = -((H11*H22 + (m*l*b*np.cos(x1-x2))**2)**(-2))*(2*m*l*b*np.cos(x1-x2))*(-m*l*b*np.sin(x1-x2))
        K_dq2 = -((H11*H22 + (m*l*b*np.cos(x1-x2))**2)**(-2))*(2*m*l*b*np.cos(x1-x2))*(m*l*b*np.sin(x1-x2))

        partials['x3_dot', 'x1'] = (x3**2)*((H12_dq1*K*h) + (H12*K_dq1*h) + (H12*K*h_dq1)) + ((x4**2)*H22)*((K_dq1*h) + (K*h_dq1)) - (H22*((K_dq1*G1) + (K*G1_dq1))) + G2*((H12_dq1*K) + (H12*K_dq1)) #- (H22*tau*(K_dq1)) - (tau*((H12_dq1*K) + (H12*K_dq1)))
        partials['x3_dot', 'x2'] = (x3**2)*((H12_dq2*K*h) + (H12*K_dq2*h) + (H12*K*h_dq2)) + ((x4**2)*H22)*((K_dq2*h) + (K*h_dq2)) - (H22*K_dq2*G1) + ((H12_dq2*K*G2) + (H12*K_dq2*G2) + (H12*K*G2_dq2)) #- (H22*tau*(K_dq2)) - (tau*((H12_dq2*K) + (H12*K_dq2)))
        partials['x3_dot', 'x3'] = 2*H12*K*h*x3
        partials['x3_dot', 'x4'] = 2*H22*K*h*x4
        #partials['x3_dot', 'tau'] = -(H22 + H12)*K

        partials['x4_dot', 'x1'] = (-x3**2)*H11*((K_dq1*h) + (K*h_dq1)) + (-x4**2)*((H12_dq1*K*h) + (H12*K_dq1*h) + (H12*K*h_dq1)) + ((H12_dq1*K*G1) + (H12*K_dq1*G1) + (H12*K*G1_dq1)) - (H11*K_dq1*G2) #+ (tau*((H12_dq1*K) + (H12*K_dq1))) + H11*tau*K_dq1
        partials['x4_dot', 'x2'] = (-x3**2)*H11*((K_dq2*h) + (K*h_dq2)) + (-x4**2)*((H12_dq2*K*h) + (H12*K_dq2*h) + (H12*K*h_dq2))+ G1*((H12_dq2*K) + (H12*K_dq2)) - (H11*((K_dq2*G2) + (K*G2_dq2))) #+ (tau*((H12_dq2*K) + (H12*K_dq2))) + H11*tau*K_dq2
        partials['x4_dot', 'x3'] = -2*H11*K*h*x3
        partials['x4_dot', 'x4'] = -2*H12*K*h*x4
        #partials['x4_dot', 'tau'] = (H11 + H12)*K
        
        partials['x1_dot', 'x1'] = 0
        partials['x1_dot', 'x2'] = 0
        partials['x1_dot', 'x3'] = 1
        partials['x1_dot', 'x4'] = 0
        #partials['x1_dot', 'tau'] = 0

        partials['x2_dot', 'x1'] = 0
        partials['x2_dot', 'x2'] = 0
        partials['x2_dot', 'x3'] = 0
        partials['x2_dot', 'x4'] = 1
       # partials['x2_dot', 'tau'] = 0 """
       



class CostFunc(om.ExplicitComponent):
    # Computes the Cost
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare("states_ref", types=dict)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x1', shape=(nn,),units='rad', desc='q1')
        self.add_input('x2', shape=(nn,),units='rad', desc='q2')
        self.add_input('tau', shape=(nn,), units='N*m', desc='input torque')
        

        self.add_output('costrate', shape=(nn,), desc='quadratic cost rate')
        
        self.declare_partials(of=['costrate'], wrt=['x1', 'x2', 'tau'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['m_H','m_t','m_s', 'tau',], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['m_H','m_t','m_s', 'tau',], method='fd', step=1e-6)

    def compute(self, inputs, outputs,):
        tau = inputs['tau']
        x1 = inputs['x1']
        x2 = inputs['x2']
        states_ref = self.options['states_ref']

        x1ref = states_ref['x1'] # reference states (final)
        x2ref = states_ref['x2']

        # distance of current states from final states
        dx1 = x1 - x1ref
        dx2 = x2-x2ref

        outputs['costrate'] = tau**2

    def compute_partials(self, inputs, partials,):
        tau = inputs['tau']
        x1 = inputs['x1']
        x2 = inputs['x2']
        states_ref = self.options['states_ref']

        x1ref = states_ref['x1'] # reference states (final)
        x2ref = states_ref['x2']

        # distance of current states from final states
        dx1 = x1 - x1ref
        dx2 = x2-x2ref
        

        partials['costrate', 'tau'] = 2*tau
        #partials['costrate', 'x1'] = 2*dx1
        #partials['costrate', 'x2'] = 2*dx2


def check_partials():
    nn = 3
    states_ref = {'x1': 10*(np.pi / 180), 'x3': 0, 'x2': 20*(np.pi / 180), 'x4': 0}
    p = om.Problem()
    p.model.add_subsystem('dynamics', system(num_nodes=nn, states_ref=states_ref),promotes=['*'])

    """p.model.set_input_defaults('L', val=1, units='m')
    p.model.set_input_defaults('a1', val=0.375, units='m')
    p.model.set_input_defaults('b1', val=0.125, units='m')
    p.model.set_input_defaults('a2', val=0.175, units='m')
    p.model.set_input_defaults('b2', val='0.325', units='m')
    p.model.set_input_defaults('m_H', val=0.5, units='kg')
    p.model.set_input_defaults('m_t', val=0.5, units='kg')
    p.model.set_input_defaults('m_s', val=0.05, units='kg')"""

    p.model.set_input_defaults('x1', val=np.random.random(nn))
    p.model.set_input_defaults('x3', val=np.random.random(nn))
    p.model.set_input_defaults('x2', val=np.random.random(nn))
    p.model.set_input_defaults('x4', val=np.random.random(nn))
    #p.model.set_input_defaults('tau', val=np.random.random(nn))


    # check partials
    p.setup(check=True)
    p.run_model()
    #om.n2(p)
    p.check_partials(compact_print=True)

if __name__ == '__main__':
    check_partials()

