import openmdao.api as om
import dymos as dm
from brachODE import Brachistochrone
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results

## solving the brach probelm with collocation

p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.options['print_results'] = False
p.driver.declare_coloring()

# trajectory
traj = p.model.add_subsystem('traj',dm.Trajectory())
phase = traj.add_phase('phase0', dm.Phase(ode_class=Brachistochrone, transcription=dm.GaussLobatto(num_segments=10)))

# vars
phase.set_time_options(fix_initial=True, duration_bounds=(.5,10))

phase.add_state('x', fix_initial=True, fix_final=True)
phase.add_state('y', fix_initial=True, fix_final=True)
phase.add_state('v', fix_initial=True, fix_final=False)

phase.add_control('theta', continuity=True, rate_continuity=True, units='deg', lower=0.01, upper=179.9)

phase.add_parameter('g', units='m/s**2', val=9.81)

# objective - minimize time
phase.add_objective('time',loc='final', scaler=10)
p.model.linear_solver = om.DirectSolver()

# problem setup
p.setup()

p['traj.phase0.t_initial'] = 0
p['traj.phase0.t_duration'] = 2

p.set_val('traj.phase0.states:x', phase.interp('x',ys=[0,10]))
p.set_val('traj.phase0.states:y', phase.interp('y',ys=[10,5]))
p.set_val('traj.phase0.states:v', phase.interp('v',ys=[0,9.9]))
p.set_val('traj.phase0.controls:theta', phase.interp('theta',ys=[5,100.5]))

# solving
dm.run_problem(p)

# results
print(p.get_val('traj.phase0.timeseries.time')[-1])


# generating the trajectory
exp_out = traj.simulate()

plot_results([('traj.phase0.timeseries.x','traj.phase0.timeseries.y','x', 'y'), ('traj.phase0.timeseries.time','traj.phase0.timeseries.theta','time','theta')],title='Brach solution',p_sol=p,p_sim=exp_out)

plt.show()
## test from laptop