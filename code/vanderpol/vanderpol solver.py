import openmdao.api as om
import dymos as dm
from vanODE import vanderpolODE

p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver(print_results = False)
p.driver.options['optimizer'] = 'IPOPT'
p.driver.declare_coloring()
#p.model.linear_solver = om.DirectSolver()

traj = dm.Trajectory()
p.model.add_subsystem('traj',subsys=traj)

t = dm.GaussLobatto(num_segments=10,order=3,compressed=True)

phase = dm.Phase(ode_class=vanderpolODE,transcription=t)
traj.add_phase(name='phase0',phase=phase)

t_final = 10
phase.set_time_options(fix_initial=True,fix_duration=True,duration_val=t_final)

phase.add_state('x0',fix_initial=True,fix_final=True,rate_source='x0dot',)
phase.add_state('x', fix_initial=True, fix_final=True,rate_source='xdot')
phase.add_state('J', fix_initial=True, fix_final=False,rate_source='Jdot')

phase.add_control('u',units=None,lower=-0.75,upper=1.0,rate_continuity=True)
phase.add_objective('J',loc='final')

p.setup(check=True)
p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = t_final

p['traj.phase0.states:x0'] = phase.interp('x0', [0, 0])
p['traj.phase0.states:x'] = phase.interp('x', [1, 0])
p['traj.phase0.states:J'] = phase.interp('J', [0, 1])
p['traj.phase0.controls:u'] = phase.interp('u', [0, 0])

dm.run_problem(p)






    