import numpy as np
import openmdao.api as om
import dymos as dm
from compassdynamics import system
from compassdynamics import transition
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

    """ 
    walker will complete one full cycle -- states will be the same at the end as they were at the beginning (maybe ?)
    """
    states_init = {'x1': 0, 'x3': 0.4, 'x2': 0, 'x4': -2}
    states_final = {'x1': 0.4, 'x3': 0, 'x2': -0.6, 'x4': 0}

    p = om.Problem()

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False
    p.driver.declare_coloring(tol=1.0E-12, orders=None)

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    lockphase = traj.add_phase('lockphase', dm.Phase(ode_class=system, transcription=dm.GaussLobatto(num_segments=20, order=3), ode_init_kwargs={'states_ref': states_final}))
    lockphase.set_time_options(fix_initial=True, initial_val=0, fix_duration=True, duration_val=duration_lockphase, units='s') # set time of simulation    \

    transphase = traj.add_phase('transphase', dm.Phase(ode_class=transition, transcription=dm.GaussLobatto(num_segments=10, order=3)))
    transphase.set_time_options(fix_initial=False, fix_duration=True, duration_val=0.1, units='s')

    #states
    lockphase.add_state('x1', fix_initial=True, upper=3, lower=-3, rate_source='x1_dot', units='rad')
    lockphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
    lockphase.add_state('x2', fix_initial=True, upper=3, lower=-3, rate_source='x2_dot', units='rad')
    lockphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
    lockphase.add_state('cost', fix_initial=True, rate_source='costrate')

    lockphase.add_control('tau', lower = 0, upper = 10, fix_initial=False, units='N*m') # add control torque

    # paramaters
    lockphase.add_parameter('a', val=a, units='m', static_target=True)
    lockphase.add_parameter('b', val=b, units='m', static_target=True)
    lockphase.add_parameter('mh', val=mh, units='kg', static_target=True)
    lockphase.add_parameter('m', val=m, units='kg', static_target=True)

    # end contraints
    #lockphase.add_boundary_constraint('x1', loc='final', equals=states_final['x1'], units='rad')
    #lockphase.add_boundary_constraint('x3', loc='final', equals=states_final['x3'], units='rad/s')
    #lockphase.add_boundary_constraint('x2', loc='final', equals=states_final['x2'], units='rad')
    #lockphase.add_boundary_constraint('x4', loc='final', equals=states_final['x4'], units='rad/s')

    # start constraints
    lockphase.add_boundary_constraint('x1', loc='initial', equals=states_init['x1'], units='rad')
    lockphase.add_boundary_constraint('x3', loc='initial', equals=states_init['x3'], units='rad/s')
    lockphase.add_boundary_constraint('x2', loc='initial', equals=states_init['x2'], units='rad')
    lockphase.add_boundary_constraint('x4', loc='initial', equals=states_init['x4'], units='rad/s')

    lockphase.add_objective('cost')

    lockphase.add_timeseries_output('alpha', output_name='alpha', units='rad', timeseries='timeseries')

    #states
    transphase.add_state('x1', fix_initial=True, rate_source='x1_new', units='rad')
    transphase.add_state('x3', fix_initial=True, rate_source='x3_new', units='rad/s')
    transphase.add_state('x2', fix_initial=True, rate_source='x2_new', units='rad')
    transphase.add_state('x4', fix_initial=True, rate_source='x4_new', units='rad/s')


    p.setup(check=True)

    """p.set_val('traj.lockphase.states:x1', lockphase.interp(ys=[states_init['x1'], states_final['x1']], nodes='state_input'), units='rad')
    p.set_val('traj.lockphase.states:x3', lockphase.interp(ys=[states_init['x3'], states_final['x3']], nodes='state_input'), units='rad/s')
    p.set_val('traj.lockphase.states:x2', lockphase.interp(ys=[states_init['x2'], states_final['x2']], nodes='state_input'), units='rad')
    p.set_val('traj.lockphase.states:x4', lockphase.interp(ys=[states_init['x4'], states_final['x4']], nodes='state_input'), units='rad/s')
    p.set_val('traj.lockphase.states:cost', lockphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
    p.set_val('traj.lockphase.controls:tau', lockphase.interp(ys=[0, 10], nodes='control_input'), units='N*m') """

    alpha = p.get_val('traj.lockphase.timeseries.alpha')


    # calculating transition matrices for phase change
    Q11_m = -m*a*b; Q12_m = -m*a*b + ((mh*l**2) + 2*m*a*l)*np.cos(2*alpha); Q22_m = Q11_m
    Q11_p = m*b*(b - l*np.cos(2*alpha)); Q12_p = m*l*(l-b*np.cos(2*alpha)) + m*a**2 + mh*l**2; Q21_p = m*b**2; Q22_p = -m*b*l*np.cos(2*alpha)

    kq_p = 1 / (Q11_p*Q22_p - Q12_p*Q21_p) # inverse constant

    # transition matrix for velocities
    P11 = kq_p*(Q22_p*Q11_m ); P12 = kq_p*(Q22_p*Q12_m - Q12_p*Q22_m); P21 = kq_p*(-Q21_p*Q11_m); P22 = kq_p*(Q11_p*Q22_m)

    traj.link_phases(phases=['lockphase', 'transphase'], vars=['x1', 'x2', 'x3', 'x4'], locs=['final', 'initial'])
    #traj.link_phases(phases=['transphase', 'lockphase'], vars=['x1', 'x2', 'x3', 'x4'], locs=['final', 'initial'])

    #traj.add_linkage_constraint('lockphase', 'lockphase', 'x1', 'x2')
    #traj.add_linkage_constraint('lockphase', 'lockphase', 'x2', 'x1')

    
    

    dm.run_problem(p, run_driver=True, simulate=True, make_plots=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10})

    #om.n2(p)

   

    # print values - since there is no objective atm this doesnt mean anything
    #print('L:', p.get_val('L', units='m'))
    # print('q1:', p.get_val('q1', units='rad'))
    # print('q1_dot:', p.get_val('q1_dot', units='rad/s'))
    # print('q2:', p.get_val('q2', units='rad'))
    # print('q2_dot:', p.get_val('q2_dot', units='rad/s'))
    cost = p.get_val('traj.lockphase.states:cost')[-1]
    print('cost: ', cost)

    sim_sol = om.CaseReader('dymos_simulation.db').get_case('final')

    # plot time history

    plot_results([('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x1','time', 'q1'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x2','time','q2'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x3','time','q1_dot'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x4','time','q2_dot'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.controls:tau','time','tau'),
                  ('traj.lockphase.timeseries.time', 'traj.lockphase.timeseries.alpha', 'time', 'alpha'),
                  ('traj.lockphase.timeseries.time', 'traj.lockphase.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p,p_sim=sim_sol)
    plt.savefig('compass_gait.pdf', bbox_inches='tight')

    x_data = p.get_val('traj.lockphase.states:x1')
    y_data = p.get_val('traj.lockphase.states:x3')

    fig, ax = plt.subplots()

    ax.plot(x_data, y_data, linewidth=2.0)

    ax.set(xlim=(-1.5, 1.3), ylim=(-5, 5))

    ax.set_xlabel('q1 angle')
    ax.set_ylabel('q1 angular velocity')

    plt.savefig('limitcycle_compass.pdf')

if __name__ == '__main__':
    main()
