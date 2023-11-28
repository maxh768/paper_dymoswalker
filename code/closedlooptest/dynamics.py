import numpy as np
import openmdao.api as om

class dynamics(om.ExplicitComponent):
    """
    goal: find feedback matrix k to make a simple system stable using optimization
    system:   x1dot = x2 + u1
              x2dot = 2x1 - x2
              x1 = output
    inputs: x2
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('u1',val=3, shape=(nn,),units='none')
        self.add_input('x1',shape=(nn,),units='none')
        self.add_input('x2',shape=(nn,),units='none')

        self.add_output('x1dot', shape=(nn,),units='none')
        self.add_output('x2dot',shape=(nn,),units='none')

        self.declare_partials(of='x1dot', wrt='x2',val=1, rows=arange, cols=arange)
        self.declare_partials(of='x2dot', wrt='x1',val=2, rows=arange, cols=arange)
        self.declare_partials(of='x2dot', wrt='x2',val=-2, rows=arange, cols=arange)
        
    def compute(inputs, outputs, self):
        u1 = inputs('u1')
        x1 = inputs('x1')
        x2 = inputs('x2')

        outputs['x1dot'] = x2+u1
        outputs['x2dot'] = 2*x1 - x2

class feedback():

    def initizize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated')

    def setup(self):
        self.add_input('x1', shape=(nn,), units='none')
        self.add_input('x2', shape=(nn,), units='none')
        self.add_input('u1',val=3, shape=(nn,), units='none')

        self.add_input('K', shape=(1, 2))
      
    