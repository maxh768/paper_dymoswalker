import numpy as np
import openmdao.api as om
import dymos as dm
from compassdynamics import system
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results
import numpy as np


def main():
    duration_lockphase = 3 # duration of locked knee phase

    # defining paramters of the legged walker

    a = 0.5
    b = 0.5
    mh = 10
    m = 5
    l = a + b
    phi = 0.0525 # slope - 3 degrees
    phi_contraint = -2*phi

    """ 
    walker will complete one full cycle -- states will be the same at the end as they were at the beginning
    """
    states_init = {'x1': 0, 'x3': 2, 'x2': 0, 'x4': -0.4} # initial conditions
    states_final = {'x1': -0.3, 'x3': 0, 'x2': -0.3, 'x4': 0} # final guess - not really used

    p = om.Problem()

    p.driver = om.pyOptSparseDriver() # set driver to ipopt
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False
    p.driver.declare_coloring(tol=1.0E-12, orders=None)

    traj = p.model.add_subsystem('traj', dm.Trajectory()) # init trajectory

    # add phase
    lockphase = traj.add_phase('lockphase', dm.Phase(ode_class=system, transcription=dm.GaussLobatto(num_segments=20, order=3), ode_init_kwargs={'states_ref': states_final}))
    lockphase.set_time_options(fix_initial=True, initial_val=0, fix_duration=True, duration_val=duration_lockphase, units='s') # set time options for simulation

    #states
    lockphase.add_state('x1', fix_initial=True, rate_source='x1_dot', units='rad')
    lockphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
    lockphase.add_state('x2', fix_initial=True, rate_source='x2_dot', units='rad')
    lockphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
    lockphase.add_state('cost', fix_initial=True, rate_source='costrate')

    lockphase.add_control('tau', lower = 0, upper = 10, fix_initial=False, units='N*m') # add control torque

    # paramaters
    lockphase.add_parameter('a', val=a, units='m', static_target=True)
    lockphase.add_parameter('b', val=b, units='m', static_target=True)
    lockphase.add_parameter('mh', val=mh, units='kg', static_target=True)
    lockphase.add_parameter('m', val=m, units='kg', static_target=True)

    # end contraints
    lockphase.add_boundary_constraint('x1', loc='final', equals=states_final['x1'], units='rad')
    #lockphase.add_boundary_constraint('x3', loc='final', equals=states_final['x3'], units='rad/s')
    lockphase.add_boundary_constraint('x2', loc='final', equals=states_final['x2'], units='rad')
    #lockphase.add_boundary_constraint('x4', loc='final', equals=states_final['x4'], units='rad/s')

    # add auxilary outputs 
    lockphase.add_timeseries_output('x3changer', output_name='x3changer', units='rad', timeseries='timeseries') 
    lockphase.add_timeseries_output('x4changer', output_name='x4changer', units='rad', timeseries='timeseries') 


    # transition boudary contraints
    #lockphase.add_boundary_constraint('phi_bounds', loc='final', equals=phi_contraint,  units='rad')
    #lockphase.add_boundary_constraint('alpha_bounds', loc='final', equals=0, units='rad')

    # start constraints
    lockphase.add_boundary_constraint('x1', loc='initial', equals=states_init['x1'], units='rad')
    lockphase.add_boundary_constraint('x3', loc='initial', equals=states_init['x3'], units='rad/s')
    lockphase.add_boundary_constraint('x2', loc='initial', equals=states_init['x2'], units='rad')
    lockphase.add_boundary_constraint('x4', loc='initial', equals=states_init['x4'], units='rad/s')

    lockphase.add_objective('cost')

    



    p.setup(check=True)

    """p.set_val('traj.lockphase.states:x1', lockphase.interp(ys=[states_init['x1'], states_final['x1']], nodes='state_input'), units='rad')
    p.set_val('traj.lockphase.states:x3', lockphase.interp(ys=[states_init['x3'], states_final['x3']], nodes='state_input'), units='rad/s')
    p.set_val('traj.lockphase.states:x2', lockphase.interp(ys=[states_init['x2'], states_final['x2']], nodes='state_input'), units='rad')
    p.set_val('traj.lockphase.states:x4', lockphase.interp(ys=[states_init['x4'], states_final['x4']], nodes='state_input'), units='rad/s')
    p.set_val('traj.lockphase.states:cost', lockphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
    p.set_val('traj.lockphase.controls:tau', lockphase.interp(ys=[0, 10], nodes='control_input'), units='N*m') """



    # phase linkage contraints with transition equations
    #traj.add_linkage_constraint('lockphase', 'lockphase', 'x1', 'x2')
    #traj.add_linkage_constraint('lockphase', 'lockphase', 'x2', 'x1')
    #traj.add_linkage_constraint('lockphase', 'lockphase', 'x3', '')

    
    

    dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10})

    #om.n2(p)



    # print cost
    cost = p.get_val('traj.lockphase.states:cost')[-1]
    print('cost: ', cost)

    sim_sol = om.CaseReader('dymos_simulation.db').get_case('final')

    # plot time history

    plot_results([('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x1','time', 'q1'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x2','time','q2'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x3','time','q1_dot'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x4','time','q2_dot'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.controls:tau','time','tau'),
                  ('traj.lockphase.timeseries.time', 'traj.lockphase.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p,p_sim=sim_sol)
    plt.savefig('compass_gait.pdf', bbox_inches='tight')

    x_data = p.get_val('traj.lockphase.states:x1')
    y_data = p.get_val('traj.lockphase.states:x3')

    fig, ax = plt.subplots()

    ax.plot(x_data, y_data, linewidth=2.0)

    ax.set(xlim=(-3, 3), ylim=(-5, 5))

    ax.set_xlabel('q1 angle')
    ax.set_ylabel('q1 angular velocity')

    plt.savefig('limitcycle_compass.pdf')

if __name__ == '__main__':
    main()
