import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import numpy as np

class vanderpolODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('x', val=np.ones(nn),desc='position of oscilitor',)
        self.add_input('x0',val=np.ones(nn),desc='rate of change of x',)
        self.add_input('u', val=np.ones(nn), desc='control',)

        self.add_output('xdot', val=np.ones(nn))
        self.add_output('x0dot', val=np.ones(nn))
        self.add_output('Jdot', val=np.ones(nn),desc='cost function')


        self.declare_partials(of='x0dot', wrt='x0')
        self.declare_partials(of='x0dot', wrt='x'  )
        self.declare_partials(of='x0dot', wrt='u',val=1.0)

        self.declare_partials(of='xdot', wrt='x0',val=1.0)

        self.declare_partials(of='Jdot', wrt='x0')
        self.declare_partials(of='Jdot', wrt='x')
        self.declare_partials(of='Jdot', wrt='u')

    def compute(self, inputs, outputs):
        x0 = inputs['x0']
        x = inputs['x']
        u = inputs['u']

        outputs['xdot'] = x0
        outputs['x0dot'] = (1 - x**2)*x0 - x + u
        outputs['Jdot'] = x0**2 + x**1 + u**2
    
    def compute_partials(self, inputs, jacobian):
        x0 = inputs['x0']
        x = inputs['x']
        u = inputs['u']

        jacobian['x0dot','x0'] = 1-x*x
        jacobian['x0dot','x'] = -2.0 * x * x0 - 1.0

        jacobian['Jdot','x0'] = 2.0 * x0
        jacobian['Jdot','x'] = 2*x
        jacobian['Jdot','u'] = 2*u
        
