import numpy as np
import openmdao.api as om
import dymos as dm
from leggeddynamics import kneedWalker
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results


def main():
    duration_lockphase = 10 # duration of locked knee phase

    # defining paramters of the legged walker
    L = 1
    a1 = 0.375
    a2 = 0.175
    b1 = 0.125
    b2 = 0.325
    m_H = 0.5
    m_t = 0.5
    m_s = 0.05

    """ 
    walker will complete one full cycle -- states will be the same at the end as they were at the beginning (maybe ?)
    """
    states_init = {'x1': 0, 'x3': 0.4, 'x2': 0, 'x4': -2}
    states_final = {'x1': -45*(np.pi / 180), 'x3': 0, 'x2': -50*(np.pi / 180), 'x4': 0}

    p = om.Problem()

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False
    p.driver.declare_coloring(tol=1.0E-12, orders=None)

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    lockphase = traj.add_phase('lockphase', dm.Phase(ode_class=kneedWalker, transcription=dm.GaussLobatto(num_segments=100, order=3), ode_init_kwargs={'states_ref': states_final}))

    lockphase.set_time_options(fix_initial=True, initial_val=0, fix_duration=True, duration_val=duration_lockphase, duration_ref=duration_lockphase, units='s') # set time of simulation    

    #states
    lockphase.add_state('x1', fix_initial=True, rate_source='x1_dot', units='rad')
    lockphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
    lockphase.add_state('x2', fix_initial=True, rate_source='x2_dot', units='rad')
    lockphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
    lockphase.add_state('cost', fix_initial=True, rate_source='costrate')

    lockphase.add_control('tau', lower = 0, upper = 10, fix_initial=False, units='N*m') # add control torque

    # paramaters
    lockphase.add_parameter('L', val=L, units='m', static_target=True)
    lockphase.add_parameter('a1', val=a1, units='m', static_target=True)
    lockphase.add_parameter('a2', val=a2, units='m', static_target=True)
    lockphase.add_parameter('b1', val=b1, units='m', static_target=True)
    lockphase.add_parameter('b2', val=b2, units='m', static_target=True)
    lockphase.add_parameter('m_H', val=m_H, units='kg', static_target=True)
    lockphase.add_parameter('m_t', val=m_t, units='kg', static_target=True)
    lockphase.add_parameter('m_s', val=m_s, units='kg', static_target=True)

    # end contraints
    lockphase.add_boundary_constraint('x1', loc='final', equals=states_final['x1'], units='rad')
    #lockphase.add_boundary_constraint('x3', loc='final', equals=states_final['x3'], units='rad/s')
    lockphase.add_boundary_constraint('x2', loc='final', equals=states_final['x2'], units='rad')
    #lockphase.add_boundary_constraint('x4', loc='final', equals=states_final['x4'], units='rad/s')

    # start constraints
    lockphase.add_boundary_constraint('x1', loc='initial', equals=states_init['x1'], units='rad')
    lockphase.add_boundary_constraint('x3', loc='initial', equals=states_init['x3'], units='rad/s')
    lockphase.add_boundary_constraint('x2', loc='initial', equals=states_init['x2'], units='rad')
    lockphase.add_boundary_constraint('x4', loc='initial', equals=states_init['x4'], units='rad/s')

    # add objective - TO DO - add cost function and obj
    lockphase.add_objective('cost')


    p.setup(check=True)

    # initial guess / time - lockphase
    p.set_val('traj.lockphase.states:x1', lockphase.interp(ys=[states_init['x1'], states_final['x1']], nodes='state_input'), units='rad')
    p.set_val('traj.lockphase.states:x3', lockphase.interp(ys=[states_init['x3'], states_final['x3']], nodes='state_input'), units='rad/s')
    p.set_val('traj.lockphase.states:x2', lockphase.interp(ys=[states_init['x2'], states_final['x2']], nodes='state_input'), units='rad')
    p.set_val('traj.lockphase.states:x4', lockphase.interp(ys=[states_init['x4'], states_final['x4']], nodes='state_input'), units='rad/s')
    p.set_val('traj.lockphase.states:cost', lockphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
    p.set_val('traj.lockphase.controls:tau', lockphase.interp(ys=[-10, 10], nodes='control_input'), units='N*m')


    # simulate and run problem
    dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10})

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
                  ('traj.lockphase.timeseries.time', 'traj.lockphase.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p,p_sim=sim_sol)
    plt.savefig('openloop_locked_kneedwalker.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()

