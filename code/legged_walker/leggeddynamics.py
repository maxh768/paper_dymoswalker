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

        input_names = ['L', 'a1', 'b1', 'a2', 'b2', 'q1', 'q2', 'q1_dot', 'q2_dot', 'm_H', 'm_t', 'm_s', 'tau']
        self.add_subsystem('lockedknee', lockedKneeDynamics(num_nodes=nn, ), promotes_inputs=input_names, promotes_outputs=['*'])
        #self.add_subsystem('threeLink', threeLinkDynamics(num_nodes=nn, ), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('cost', CostFunc(num_nodes=nn, states_ref=self.options['states_ref'] ), promotes_inputs=['*'], promotes_outputs=['*'])

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

        # q1 and q2
        self.add_input('q1', shape=(nn,),units='rad', desc='q1 angle')
        self.add_input('q2', shape=(nn,),units='rad', desc='q2 angle')
        self.add_input('q1_dot', shape=(nn,),units='rad/s', desc='q1 angular velocity')
        self.add_input('q2_dot', shape=(nn,),units='rad/s', desc='q2 anglular velocity')

        # masses
        self.add_input('m_H', shape=(1,),units='kg', desc='hip mass')
        self.add_input('m_t', shape=(1,),units='kg', desc='thigh mass')
        self.add_input('m_s', shape=(1,),units='kg', desc='shank mass')

        # applied torque
        self.add_input('tau', shape=(nn,),units='N*m', desc='applied toruqe at hip')


        # outputs
        # angular accelerations of q1 and q1 (state rates)
        self.add_output('q1_dotdot', shape=(nn,), units='rad/s**2',desc='angular acceleration of q1')
        self.add_output('q2_dotdot', shape=(nn,), units='rad/s**2',desc='angular acceleration of q2')

        #partials
        self.declare_partials(of=['*'], wrt=['q1', 'q1_dot', 'q2', 'q2_dot', 'tau'], method='cs')#, rows=np.arange(nn), cols=np.arange(nn))
        self.declare_coloring(wrt=['q1', 'q1_dot', 'q2','q2_dot'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['q1', 'q1_dot', 'q2','q2_dot'], method='fd', step=1e-6)

        self.declare_partials(of=['*'], wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs')# rows=np.arange(nn), cols=np.arange(nn))
        self.declare_coloring(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='fd', step=1e-6)

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
        q1 = inputs['q1']
        q2 = inputs['q2']
        q1_dot = inputs['q1_dot']
        q2_dot = inputs['q2_dot']
        tau = inputs['tau']

        ls = a1 + b1
        lt = a2 + b2

        # mtrix components
        H11 = m_s*a1**2 + m_t*(ls + a2)**2 + (m_H + m_s + m_t)*(L**2)
        H12 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(q2-q1)) #
        H22 = m_t*b2**2 + m_s*(lt + b1)**2
        h = -(m_t*b2 + m_s*(lt + b1))*(L*np.sin(q1-q2)) #
        G1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.sin(q1) #
        G2 = (m_t*b2 + m_s*(lt + b1))*g*np.sin(q2) #

        K = 1 / (H11*H22 - H12*H12) # inverse constant

        outputs['q1_dotdot'] = H12*K*h*q1_dot**2 + H22*K*h*q2_dot**2 - (H22*K*G1 - H12*K*G2) + ((-H22*K - H12*K)*tau)
        outputs['q2_dotdot'] = -H11*K*h*q1_dot**2 - H12*K*h*q2_dot**2 - (-H12*K*G1 + H11*K*G2) + ((H12*K + H11*K)*tau)

    def compute_partials(self, inputs, partials):
        """
        this does not do anything at the moment

        """

        g = self.options['g']
        L = inputs['L']
        a1 = inputs['a1']
        a2 = inputs['a2']
        b1 = inputs['b1']
        b2 = inputs['b2']
        m_H = inputs['m_H']
        m_t = inputs['m_t']
        m_s = inputs['m_s']
        q1 = inputs['q1']
        q2 = inputs['q2']
        q1_dot = inputs['q1_dot']
        q2_dot = inputs['q2_dot']
        tau = inputs['tau']

        ls = a1 + b1
        lt = a2 + b2

        H11 = m_s*a1**2 + m_t*(ls + a2)**2 + (m_H + m_s + m_t)*(L**2)
        H12 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(q2-q1)) #
        H22 = m_t*b2**2 + m_s*(lt + b1)**2
        h = -(m_t*b2 + m_s*(lt + b1))*(L*np.sin(q1-q2)) #
        G1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.sin(q1) #
        G2 = (m_t*b2 + m_s*(lt + b1))*g*np.sin(q2)

        K = 1 / (H11*H22 - H12*H12)

        # partials of terms wrt q1 q2
        H12_dq1 = -(m_t*b2 + m_s*(lt + b1))*L*np.sin(q2-q1)
        H12_dq2 = -(m_t*b2 + m_s*(lt + b1))*L*np.sin(q2-q1)*(-1)
        h_dq1 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(q1-q2))
        h_dq2 = -(m_t*b2 + m_s*(lt + b1))*(L*np.cos(q1-q2))*(-1)
        G1_dq1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_t + m_s)*L)*g*np.cos(q1)
        G2_dq2 =  (m_t*b2 + m_s*(lt + b1))*g*np.cos(q2)
            

        #partials['q1_dotdot', 'q1'] = K*q1_dot*q1_dot*(H12*h_dq1 + h*H12_dq1) + H22*K*h_dq1*q2_dot*q2_dot - (H22*K*G1_dq1) + (-H12_dq1*K*tau)
        # jacobian['q1_dotdot', 'q2'] = 


"""
3 link dynamics uses q1 q2 and q3
"""
class threeLinkDynamics(om.ExplicitComponent):
    # phase with three link dynamics
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref')
        self.options.declare('g', default=9.81, desc='gravity const')

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('L', shape=(1,),units='m',desc="length of one leg") 

        # length paramters
        self.add_input('a1',shape=(1,),units='m',desc='shank length below point mass')
        self.add_input('b1', shape=(1,),units='m', desc='shank length above point mass')

        self.add_input('a2', shape=(1,),units='m', desc='thigh length below points mass')
        self.add_input('b2', shape=(1,),units='m', desc='thigh length above point mass')

        # q1, q2, and q3
        self.add_input('q1', shape=(nn,),units='rad', desc='q1 angle')
        self.add_input('q2', shape=(nn,),units='rad', desc='q2 angle')
        self.add_input('q1_dot', shape=(nn,),units='rad/s', desc='q1 angular velocity')
        self.add_input('q2_dot', shape=(nn,),units='rad/s', desc='q2 anglular velocity')
        self.add_input('q3', shape=(nn,), units='rad', desc='q3 angle')
        self.add_input('q3_dot', shape=(nn,), units='rad/s', desc='q3 anglular velocity')
        

        # masses
        self.add_input('m_H', shape=(1,),units='kg', desc='hip mass')
        self.add_input('m_t', shape=(1,),units='kg', desc='thigh mass')
        self.add_input('m_s', shape=(1,),units='kg', desc='shank mass')

        # applied torque
        self.add_input('tau', shape=(nn,),units='N*m', desc='applied toruqe at hip')

        # outputs
        # angular accelerations of q1 q2 q3 (state rates)
        self.add_output('q1_dotdot', shape=(nn,), units='rad/s**2',desc='angular acceleration of q1')
        self.add_output('q2_dotdot', shape=(nn,), units='rad/s**2',desc='angular acceleration of q2')
        self.add_output('q3_dotdot', shape=(nn,), units='rad/s**2', desc='angular acc of q3')

        ## partials
        self.declare_partials(of=['*'], wrt=['q1', 'q1_dot', 'q2', 'q2_dot', 'tau', 'q3', 'q3_dot'], method='cs')#, rows=np.arange(nn), cols=np.arange(nn))
        self.declare_coloring(wrt=['q1', 'q1_dot', 'q2','q2_dot', 'q3', 'q3_dot'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['q1', 'q1_dot', 'q2','q2_dot', 'q3', 'q3_dot'], method='fd', step=1e-6)

        self.declare_partials(of=['*'], wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs')# rows=np.arange(nn), cols=np.arange(nn))
        self.declare_coloring(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='fd', step=1e-6)
    

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
        q1 = inputs['q1']
        q2 = inputs['q2']
        q3 = inputs['q3']
        q1_dot = inputs['q1_dot']
        q2_dot = inputs['q2_dot']
        q3_dot = inputs['q3_dot']
        tau = inputs['tau']
        ##

        ls = a1 + b1
        lt = a2 + b2
        H11 = m_s*a1**2 + m_t*(ls + a2)**2 + (m_H + m_s + m_t)*L**2
        H12 = -(m_t*b2 + m_s*lt)*L*np.cos(q2 - q1)
        H13 = -m_s*b1*L*np.cos(q3-q1)
        H22 = m_t*b2**2 + m_s*lt**2
        H23 = m_s*lt*b1*np.cos(q3-q2)
        H33 = m_s*b1**2 # components of symmetric H matrix

        h122 = -(m_t*b2 + m_s*lt)*L*np.sin(q1-q2) # components of B matrix
        h133 = -m_s*b1*L*np.sin(q1-q3)
        h211 = -h122
        h233 = m_s*lt*b1*np.sin(q3-q2)
        h311 = -h133
        h322 = -h233

        G1 = -(m_s*a1 + m_t*(ls+a2) + (m_H + m_s + m_t)*L)*g*np.sin(q1)
        G2 = (m_t*b2 + m_s*lt)*g*np.sin(q2)
        G3 = m_s*b1*g*np.sin(q3)

        DH = H11*(H22*H33-H23**2) + H12*(H13*H23 - H12*H33) + H13*(H12*H23 - H22*H13) # determinite of matrix H

        # components of adjoint of H matrix
        A11 = H22*H33 - H23*H23; A12 = -(H12*H33 - H23*H13); A13 = H12*H23 - H13*H22; A21 = A12; A22 = H11*H33 - H13*H13; A23 = -(H11*H23 - H12*H13)
        A31 = A13; A32 = A23; A33 = H11*H22 - H12*H12

        # calcuate rates
        outputs['q1_dotdot'] = (-((A12*h211*q1_dot + A13*h311*q1_dot)*q1_dot + (A11*h122*q2_dot + A13*h322*q2_dot)*q2_dot + (A11*h133*q3_dot + A12*h233*q3_dot)*q3_dot + A11*G1 + A12*G2 + A13*G3) - A11 + A12 + A13)*(1/DH)
        outputs['q2_dotdot'] = (-((A22*h211*q1_dot + A23*h311*q1_dot)*q1_dot + (A21*h122*q2_dot + A23*h322*q2_dot)*q2_dot + (A21*h133*q3_dot + A22*h233*q3_dot)*q3_dot + A21*G1 + A22*G2 + A23*G3) + -A21 + A22 + A23)*(1/DH)
        outputs['q3_dotdot'] = -(((A32*h211*q1_dot + A33*h311*q1_dot)*q1_dot + (A31*h122*q2_dot + A33*h322*q2_dot)*q2_dot + (A31*h133*q3_dot + A32*h233*q3_dot)*q3_dot +A31*G1 + A32*G2 + A33*G3) + -A31 + A32 + A33)*(1/DH)

        """ H = np.array([[H11, H12, H13], [H12, H22, H23], [H13, H23, H33]], dtype=float)
        B = np.array([[0, h122*q2_dot, h133*q3_dot],[h211*q1_dot, 0, h233*q3_dot], [h311*q1_dot, h322*q2_dot, 0]], dtype=float)
        G = np.array([[G1], [G2], [G3]], ddtype=float)
        R = np.array([[-1], [1], [1]], dtype=float) # control matrix, come back and check...

        qdotv = np.array([[q1_dot], [q2_dot], [q3_dot]], dtype=float)
        
        hinv = inv(H)

        qvector = -(multi_dot(B, qdotv, hinv)) - np.dot(G, hinv) + np.dot(R, hinv)*tau

        outputs['q1_dotdot'] = qvector[0]
        outputs['q2_dotdot'] = qvector[1]
        outputs['q3_dotdot'] = qvector[2]   """

    def computs_partials(self, inputs, outputs):
        """ this does not do anything """

        

        g = self.options['g']
        L = inputs['L']
        a1 = inputs['a1']
        a2 = inputs['a2']
        b1 = inputs['b1']
        b2 = inputs['b2']
        m_H = inputs['m_H']
        m_t = inputs['m_t']
        m_s = inputs['m_s']
        q1 = inputs['q1']
        q2 = inputs['q2']
        q3 = inputs['q3']
        q1_dot = inputs['q1_dot']
        q2_dot = inputs['q2_dot']
        q3_dot = inputs['q3_dot']
        tau = inputs['tau']

class CostFunc(om.ExplicitComponent):
    # Computes the Cost
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare("states_ref", types=dict)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('m_H', shape=(1,),units='kg', desc='hip mass')
        self.add_input('m_t', shape=(1,),units='kg', desc='thigh mass')
        self.add_input('m_s', shape=(1,),units='kg', desc='shank mass')
        self.add_input('tau', shape=(nn,), units='N*m', desc='input torque')
        

        self.add_output('costrate', shape=(nn,), desc='quadratic cost rate')
        
        self.declare_partials(of=['costrate'], wrt=['m_H', 'm_t', 'm_s', 'tau',], method='cs')
        self.declare_coloring(wrt=['m_H','m_t','m_s', 'tau',], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['m_H','m_t','m_s', 'tau',], method='fd', step=1e-6)

    def compute(self, inputs, outputs,):
        tau = inputs['tau']
        m_H = inputs['m_H']
        m_t = inputs['m_t']
        m_s = inputs['m_s']


        m_total = m_H + m_t + m_s

        outputs['costrate'] = m_total**2 + tau**2

    def compute_partials(self, inputs, partials,):
        # doesnt do anything
        m_H = inputs['m_H']
        
   

def check_partials():
    nn = 3
    states_ref = {'q1': 10*(np.pi / 180), 'q1_dot': 0, 'q2': 20*(np.pi / 180), 'q2_dot': 0}
    p = om.Problem()
    p.model.add_subsystem('dynamics', threeLinkDynamics(num_nodes=nn, states_ref=states_ref),promotes=['*'])

    p.model.set_input_defaults('L', val=1, units='m')
    p.model.set_input_defaults('a1', val=0.375, units='m')
    p.model.set_input_defaults('b1', val=0.125, units='m')
    p.model.set_input_defaults('a2', val=0.175, units='m')
    p.model.set_input_defaults('b2', val='0.325', units='m')
    p.model.set_input_defaults('m_H', val=0.5, units='kg')
    p.model.set_input_defaults('m_t', val=0.5, units='kg')
    p.model.set_input_defaults('m_s', val=0.05, units='kg')

    p.model.set_input_defaults('q1', val=np.random.random(nn))
    p.model.set_input_defaults('q1_dot', val=np.random.random(nn))
    p.model.set_input_defaults('q2', val=np.random.random(nn))
    p.model.set_input_defaults('q2_dot', val=np.random.random(nn))
    p.model.set_input_defaults('tau', val=np.random.random(nn))
    p.model.set_input_defaults('q3', val=np.random.random(nn))
    p.model.set_input_defaults('q3_dot', val=np.random.random(nn))


    # check partials
    p.setup(check=True)
    p.run_model()
    #om.n2(p)
    p.check_partials(compact_print=True)

if __name__ == '__main__':
    check_partials()
    