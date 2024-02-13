""" dynamics of a legged walker robot"""

import numpy as np
import openmdao.api as om
from numpy.linalg import inv
from numpy.linalg import multi_dot

class kneedWalker(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)

    def setup(self):
        nn=self.options['num_nodes']

        input_names = ['L', 'a1', 'b1', 'a2', 'b2', 'x1', 'x2', 'x3', 'x4', 'm_H', 'm_t', 'm_s']
        self.add_subsystem('lockedknee', lockedKneeDynamics(num_nodes=nn, ), promotes_inputs=input_names, promotes_outputs=['*'])
        self.add_subsystem('cost', CostFunc(num_nodes=nn, states_ref=self.options['states_ref'] ), promotes_inputs=['x1', 'x2', 'tau'], promotes_outputs=['*'])



class lockedKneeDynamics(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)
        self.options.declare('g', default=9.81, desc='gravity constant')

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('L', shape=(1,),units='m',desc="length of one leg") 

        # length paramters
        self.add_input('a1',shape=(1,),units='m',desc='shank length below point mass')
        self.add_input('b1', shape=(1,),units='m', desc='shank length above point mass')

        self.add_input('a2', shape=(1,),units='m', desc='thigh length below points mass')
        self.add_input('b2', shape=(1,),units='m', desc='thigh length above point mass')

        """
        x1 = q1, x2 = q2, x3 = q1_dot, x4 = q2_dot
        """
        self.add_input('x1', shape=(nn,),units='rad', desc='q1')
        self.add_input('x2', shape=(nn,),units='rad', desc='q2')
        self.add_input('x3', shape=(nn,),units='rad/s', desc='q1 dot')
        self.add_input('x4', shape=(nn,),units='rad/s', desc='q2 dot')

        # masses
        self.add_input('m_H', shape=(1,),units='kg', desc='hip mass')
        self.add_input('m_t', shape=(1,),units='kg', desc='thigh mass')
        self.add_input('m_s', shape=(1,),units='kg', desc='shank mass')

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

        #partials
        self.declare_partials(of=['*'], wrt=['x1', 'x2', 'x3', 'x4'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['x1', 'x2', 'x3', 'x4'], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['x1', 'x2', 'x3', 'x4'], method='fd', step=1e-6)

        #self.declare_partials(of=['*'], wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs')# rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        g = self.options['g']
        L = inputs['L']
        a1 = inputs['a1']
        a2 = inputs['a2']
        b1 = inputs['b1']
        b2 = inputs['b2']
        m_H = inputs['m_H']
        m_t = inputs['m_t']
        m_s = inputs['m_s']
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        #tau = inputs['tau']

        ls = a1 + b1
        lt = a2 + b2

        # mtrix components
        H11 = m_s*a1**2 + m_t*(ls + a2)**2 + (m_H + m_s + m_t)*(L**2)
        H12 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(x2-x1)) #
        H22 = m_t*b2**2 + m_s*(lt + b1)**2
        h = -(m_t*b2 + m_s*(lt + b1))*(L*np.sin(x1-x2)) #
        G1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.sin(x1) #
        G2 = (m_t*b2 + m_s*(lt + b1))*g*np.sin(x2) #

        K = 1 / (H11*H22 - H12*H12) # inverse constant

        outputs['x1_dot'] = x3
        outputs['x2_dot'] = x4
        outputs['x3_dot'] = -H12*K*h*(x3**2) + -H22*K*h*(x4**2) - (H22*K*G1 - H12*K*G2) #- ((H22 + H12)*tau*K)
        outputs['x4_dot'] = H11*K*h*(x3**2) + H12*K*h*(x4**2) - (-H12*K*G1 + H11*K*G2) #+ ((H12*K + H11*K)*tau)

    def compute_partials(self, inputs, partials):
       # computes analytical partials 

        g = self.options['g']
        L = inputs['L']
        a1 = inputs['a1']
        a2 = inputs['a2']
        b1 = inputs['b1']
        b2 = inputs['b2']
        m_H = inputs['m_H']
        m_t = inputs['m_t']
        m_s = inputs['m_s']
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        #tau = inputs['tau']

        ls = a1 + b1
        lt = a2 + b2

        H11 = m_s*a1**2 + m_t*(ls + a2)**2 + (m_H + m_s + m_t)*(L**2)
        H12 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(x2-x1)) #
        H22 = m_t*b2**2 + m_s*(lt + b1)**2
        h = -(m_t*b2 + m_s*(lt + b1))*(L*np.sin(x1-x2)) #
        G1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.sin(x1) #
        G2 = (m_t*b2 + m_s*(lt + b1))*g*np.sin(x2)

        K = 1 / (H11*H22 - H12*H12)

        # partials of terms wrt q1 q2
        H12_dq1 = -(m_t*b2 + m_s*(lt + b1))*L*np.sin(x2-x1)
        H12_dq2 = -(m_t*b2 + m_s*(lt + b1))*L*np.sin(x2-x1)*(-1)
        h_dq1 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(x1-x2))
        h_dq2 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(x1-x2))*(-1)
        G1_dq1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.cos(x1)
        G2_dq2 =  (m_t*b2 + m_s*(lt + b1))*g*np.cos(x2)

        H12_abs = -H12

        K_dq1 = (-(H11*H22 + H12_abs)**(-2))*(2*H12_abs)*(-H12_dq1)
        K_dq2 = (-(H11*H22 + H12_abs)**(-2))*(2*H12_abs)*(-H12_dq2)

        partials['x3_dot', 'x1'] = (-x3**2)*((H12_dq1*K*h) + (H12*K_dq1*h) + (H12*K*h_dq1)) + ((x4**2)*H22)*((K_dq1*h) + (K*h_dq1)) - (H22*((K_dq1*G1) + (K*G1_dq1))) + G2*((H12_dq1*K) + (H12*K_dq1)) #- (H22*tau*(K_dq1)) - (tau*((H12_dq1*K) + (H12*K_dq1)))
        partials['x3_dot', 'x2'] = (-x3**2)*((H12_dq2*K*h) + (H12*K_dq2*h) + (H12*K*h_dq2)) + ((x4**2)*H22)*((K_dq2*h) + (K*h_dq2)) - (H22*K_dq2*G1) + ((H12_dq2*K*G2) + (H12*K_dq2*G2) + (H12*K*G2_dq2)) #+ (tau*((H12_dq2*K) + (H12*K_dq2))) + (tau*H11*K_dq2)
        partials['x3_dot', 'x3'] = -2*H12*K*h*x3
        partials['x3_dot', 'x4'] = -2*H22*K*h*x4
        #partials['x3_dot', 'tau'] = -(H22 + H12)*K

        partials['x4_dot', 'x1'] = (x3**2)*H11*((K_dq1*h) + (K*h_dq1)) + (-x4**2)*((H12_dq1*K*h) + (H12*K_dq1*h) + (H12*K*h_dq1)) + ((H12_dq1*K*G1) + (H12*K_dq1*G1) + (H12*K*G1_dq1)) - (H11*K_dq1*G2)
        partials['x4_dot', 'x2'] = (x3**2)*H11*((K_dq2*h) + (K*h_dq2)) + (-x4**2)*((H12_dq2*K*h) + (H12*K_dq2*h) + (H12*K*h_dq2))+ G1*((H12_dq2*K) + (H12*K_dq2)) - (H11*((K_dq2*G2) + (K*G2_dq2)))
        partials['x4_dot', 'x3'] = 2*H11*K*h*x3
        partials['x4_dot', 'x4'] = 2*H12*K*h*x4
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
        #partials['x2_dot', 'tau'] = 0

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

        outputs['costrate'] = dx1**2 + dx2**2 + tau**2

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
        partials['costrate', 'x1'] = 2*dx1
        partials['costrate', 'x2'] = 2*dx2
        

        
def check_partials():
    nn = 3
    states_ref = {'x1': 10*(np.pi / 180), 'x3': 0, 'x2': 20*(np.pi / 180), 'x4': 0}
    p = om.Problem()
    p.model.add_subsystem('dynamics', kneedWalker(num_nodes=nn, states_ref=states_ref),promotes=['*'])

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
    p.model.set_input_defaults('tau', val=np.random.random(nn))


    # check partials
    p.setup(check=True)
    p.run_model()
    #om.n2(p)
    p.check_partials(compact_print=True)

if __name__ == '__main__':
    check_partials()
    