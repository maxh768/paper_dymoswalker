import numpy as np
import openmdao.api as om

class Brachistochrone(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('static_gravity', types=(bool,), default=False, desc="If True, gravity is a scalar input rather than having different values at each node")

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('v', val=np.zeros(nn), desc="velocity", units='m/s')

        if self.options['static_gravity']:
            self.add_input('g', val=9.81, desc='gravity', units='m/s/s', tags=['dymos.static_target'])
        else: 
            self.add_input('g', val=9.81*np.ones(nn), desc='gravity', units='m/s/s')

        self.add_input('theta', val=np.ones(nn), desc='angle', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='x velocity', units='m/s', tags=['dymos.state_rate_source:x', 'dymos.state_rate_units:m'])
        self.add_output('ydot', val=np.zeros(nn), desc='y velocity',units='m/s',tags=['dymos.state_rate_source:y', 'dymos.state_rate_units:m'])
        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magn.', units='m/s**2',tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])
        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta)=constant', units='m/s')

        # partials

        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)
        if self.options['static_gravity']:
            c = np.zeros(self.options['num_nodes'])
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=c)
        else:
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)

        


                        

