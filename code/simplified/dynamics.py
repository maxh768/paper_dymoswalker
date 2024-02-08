import numpy as np
import openmdao.api as om

class system(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)

    def setup(self):
        nn=self.options['num_nodes']

        input_names = ['tau', 'x1', 'x2', 'm', 'l']
        self.add_subsystem('dynamics', dynamics(num_nodes=nn, ), promotes_inputs=input_names, promotes_outputs=['*'])
        self.add_subsystem('cost', CostFunc(num_nodes=nn, states_ref=self.options['states_ref'] ), promotes_inputs=['*'], promotes_outputs=['*'])



class dynamics(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('states_ref', types=dict)
        self.options.declare('g', default=9.81, desc='gravity constant')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('l', shape=(1,), units='m', desc='length of pendulum')
        self.add_input('m', shape=(1,), units='kg', desc='mass of pendulum')

        self.add_input('x1', shape=(nn,), units='rad', desc='angle of pendulum')
        self.add_input('x2', shape=(nn,), units='rad/s', desc='angular velocity of pendulum')

        self.add_input('tau', shape=(nn,), units='N*m', desc='inputs torque')

        self.add_output('x1_dot',shape=(nn,), units='rad/s', desc='angular velocity, same as x2')
        self.add_output('x2_dot',shape=(nn,), units='rad/s**2', desc='angular acceleration')

        self.declare_partials(of=['x1_dot', 'x2_dot'], wrt=['x1', 'x2', 'tau'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['x1', 'x2', 'tau'], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['x1', 'x2', 'tau'], method='fd', step=1e-6)
        

    def compute(self, inputs, outputs):
        l = inputs['l']
        m = inputs['m']
        x1 = inputs['x1']
        x2 = inputs['x2']
        tau = inputs['tau']
        g = self.options['g']

        outputs['x1_dot'] = x2
        outputs['x2_dot'] = -(g/l)*np.sin(x1) + (tau/(m*l**2))

    def compute_partials(self, inputs, partials):
        l = inputs['l']
        m = inputs['m']
        x1 = inputs['x1']
        x2 = inputs['x2']
        tau = inputs['tau']
        g = self.options['g']

        partials['x1_dot', 'x1'] = 0
        partials['x1_dot', 'x2'] = 1
        partials['x1_dot', 'tau'] = 0

        partials['x2_dot', 'x1'] = -(g/l)*np.cos(x1)
        partials['x2_dot', 'x2'] = 0
        partials['x2_dot', 'tau'] = 1/(m*l**2)
        
        

class CostFunc(om.ExplicitComponent):
        # Computes the Cost
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare("states_ref", types=dict)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x1', shape=(nn,),units='rad', desc='x1')
        self.add_input('x2', shape=(nn,),units='rad/s', desc='x2')
        self.add_input('tau', shape=(nn,), units='N*m', desc='input torque')
        

        self.add_output('costrate', shape=(nn,), desc='quadratic cost rate')
        
        self.declare_partials(of=['costrate'], wrt=['x1', 'x2', 'tau'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        #self.declare_coloring(wrt=['m_H','m_t','m_s', 'tau',], method='cs', show_summary=False)
        #self.set_check_partial_options(wrt=['m_H','m_t','m_s', 'tau',], method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        tau = inputs['tau']
        x1 = inputs['x1']
        x2 = inputs['x2']
        states_ref = self.options['states_ref']

        x1ref = states_ref['x1'] # reference states (final)
        x2ref = states_ref['x2']

        dx1 = x1 - x1ref
        dx2 = x2 - x2ref

        outputs['costrate'] = dx1**2 + dx2**2

    def compute_partials(self, inputs, partials):
        tau = inputs['tau']
        x1 = inputs['x1']
        x2 = inputs['x2']
        states_ref = self.options['states_ref']

        x1ref = states_ref['x1'] # reference states (final)
        x2ref = states_ref['x2']

        dx1 = x1 - x1ref
        dx2 = x2 - x2ref

        partials['costrate', 'tau'] = 0#2*tau
        partials['costrate', 'x1'] = 2*dx1
        partials['costrate', 'x2'] = 2*dx2


def check_partials():
    nn = 3
    states_ref = {'x1': 0, 'x2':0}
    p = om.Problem()
    p.model.add_subsystem('system', system(num_nodes=nn, states_ref=states_ref),promotes=['*'])

    """p.model.set_input_defaults('L', val=1, units='m')
    p.model.set_input_defaults('a1', val=0.375, units='m')
    p.model.set_input_defaults('b1', val=0.125, units='m')
    p.model.set_input_defaults('a2', val=0.175, units='m')
    p.model.set_input_defaults('b2', val='0.325', units='m')
    p.model.set_input_defaults('m_H', val=0.5, units='kg')
    p.model.set_input_defaults('m_t', val=0.5, units='kg')
    p.model.set_input_defaults('m_s', val=0.05, units='kg')"""

    p.model.set_input_defaults('x1', val=np.random.random(nn))
    p.model.set_input_defaults('x2', val=np.random.random(nn))
    p.model.set_input_defaults('tau', val=np.random.random(nn))

    # check partials
    p.setup(check=True)
    p.run_model()
    #om.n2(p)
    p.check_partials(compact_print=True)

if __name__ == '__main__':
    check_partials()
