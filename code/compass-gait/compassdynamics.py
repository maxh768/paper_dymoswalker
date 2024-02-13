import numpy as np
import openmdao.api as om
from numpy.linalg import inv
from numpy.linalg import multi_dot

class system(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)

    def setup(self):
        nn=self.options['num_nodes']

        input_names = ['a', 'b', 'x1', 'x2', 'x3', 'x4', 'mh', 'm', 'tau']
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
        self.add_input('tau', shape=(nn,),units='N*m', desc='applied toruqe at hip')


        # outputs
        """
        from state space eqns
        """
        self.add_output('x1_dot', shape=(nn,), units='rad/s',desc='q1 dot')
        self.add_output('x2_dot', shape=(nn,), units='rad/s',desc='q2 dot')
        self.add_output('x3_dot', shape=(nn,), units='rad/s**2',desc='q1 dotdot')
        self.add_output('x4_dot', shape=(nn,), units='rad/s**2',desc='q2 dotdot')

        #partials
        self.declare_partials(of=['*'], wrt=['x1', 'x2', 'x3', 'x4', 'tau'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['x1', 'x2', 'x3', 'x4'], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['x1', 'x2', 'x3', 'x4'], method='fd', step=1e-6)

        #self.declare_partials(of=['*'], wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs')# rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['a1', 'L', 'b1','a2','b2','m_H','m_t','m_s'], method='fd', step=1e-6)

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
        tau = inputs['tau']

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

        outputs['x1_dot'] = x3
        outputs['x2_dot'] = x4
        outputs['x3_dot'] = (H12*K*h*x3**2) + (H22*K*h*x4**2) - H22*K*G1 + H12*K*G2 - (H22 + H12)*K*tau
        outputs['x4_dot'] = (-H11*K*h*x3**2) - (H12*K*h*x4**2) + H12*K*G1 - H11*K*G1 + ((H12 + H11)*K*tau)

    def compute_partials(self, inputs, partials):
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
        tau = inputs['tau']

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

        H12_abs = -H12

        K_dq1 = (-(H11*H22 + H12_abs)**(-2))*(-2*H12_abs)*(-H12_dq1)
        K_dq2 = (-(H11*H22 + H12_abs)**(-2))*(-2*H12_abs)*(-H12_dq2)

        partials['x3_dot', 'x1'] = (x3**2)*((H12_dq1*K*h) + (H12*K_dq1*h) + (H12*K*h_dq1)) + ((x4**2)*H22)*((K_dq1*h) + (K*h_dq1)) - (H22*((K_dq1*G1) + (K*G1_dq1))) + G2*((H12_dq1*K) + (H12*K_dq1)) - (H22*tau*(K_dq1)) - (tau*((H12_dq1*K) + (H12*K_dq1)))
        partials['x3_dot', 'x2'] = (x3**2)*((H12_dq2*K*h) + (H12*K_dq2*h) + (H12*K*h_dq2)) + ((x4**2)*H22)*((K_dq2*h) + (K*h_dq2)) - (H22*K_dq2*G1) + ((H12_dq2*K*G2) + (H12*K_dq2*G2) + (H12*K*G2_dq2)) + (tau*((H12_dq2*K) + (H12*K_dq2))) + (tau*H11*K_dq2)
        partials['x3_dot', 'x3'] = 2*H12*K*h*x3
        partials['x3_dot', 'x4'] = 2*H22*K*h*x4
        partials['x3_dot', 'tau'] = -(H22 + H12)*K

        partials['x4_dot', 'x1'] = (-x3**2)*H11*((K_dq1*h) + (K*h_dq1)) + (-x4**2)*((H12_dq1*K*h) + (H12*K_dq1*h) + (H12*K*h_dq1)) + ((H12_dq1*K*G1) + (H12*K_dq1*G1) + (H12*K*G1_dq1)) - (H11*K_dq1*G2)
        partials['x4_dot', 'x2'] = (-x3**2)*H11*((K_dq2*h) + (K*h_dq2)) + (-x4**2)*((H12_dq2*K*h) + (H12*K_dq2*h) + (H12*K*h_dq2))+ G1*((H12_dq2*K) + (H12*K_dq2)) - (H11*((K_dq2*G2) + (K*G2_dq2)))
        partials['x4_dot', 'x3'] = -2*H11*K*h*x3
        partials['x4_dot', 'x4'] = -2*H12*K*h*x4
        partials['x4_dot', 'tau'] = (H11 + H12)*K
        
        partials['x1_dot', 'x1'] = 0
        partials['x1_dot', 'x2'] = 0
        partials['x1_dot', 'x3'] = 1
        partials['x1_dot', 'x4'] = 0
        partials['x1_dot', 'tau'] = 0

        partials['x2_dot', 'x1'] = 0
        partials['x2_dot', 'x2'] = 0
        partials['x2_dot', 'x3'] = 0
        partials['x2_dot', 'x4'] = 1
        partials['x2_dot', 'tau'] = 0
       

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

