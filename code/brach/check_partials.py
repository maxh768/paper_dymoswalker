import numpy as np
import openmdao.api as om

from brachODE import Brachistochrone

num_nodes = 5

p = om.Problem(model=om.Group())

ivc = p.model.add_subsystem('vars', om.IndepVarComp())
ivc.add_output('v', shape=(num_nodes,),units='m/s')
ivc.add_output('theta', shape=(num_nodes,),units='deg')

p.model.add_subsystem('ode', Brachistochrone(num_nodes=num_nodes))

p.model.connect('vars.v', 'ode.v')
p.model.connect('vars.theta','ode.theta')

p.setup(force_alloc_complex=True)

p.set_val('vars.v', 10*np.random.random(num_nodes))
p.set_val('vars.theta', 10*np.random.uniform(1,179, num_nodes))

p.run_model()
cpd = p.check_partials(method='cs', compact_print=True)
