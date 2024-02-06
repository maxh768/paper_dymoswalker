import numpy as np
import openmdao.api as om
import dymos as dm
from leggeddynamics import kneedWalker
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results


def main():

    duration_threelink = 12  # duration of three link dynamics phase
    duration_lockphase = 3 # duration of locked knee phase

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
    states_init = {'q1': 10*(np.pi / 180), 'q1_dot': 0, 'q2': 20*(np.pi / 180), 'q2_dot': 0, 'q3': 20*(np.pi / 180), 'q3_dot': 0}
    midphase_guess = {'q1': -10*(np.pi / 180), 'q1_dot': -5, 'q2': -20*(np.pi / 180), 'q2_dot': -5, 'q3': -20*(np.pi / 180), 'q3_dot': -5}
    states_final = {'q1': -20*(np.pi / 180), 'q1_dot': 0, 'q2': -45*(np.pi / 180), 'q2_dot': 0, 'q3': -45*(np.pi / 180), 'q3_dot': 0}

    p = om.Problem()

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False
    p.driver.declare_coloring(tol=1.0E-12, orders=None)

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    threelink = traj.add_phase('threelink', dm.Phase(ode_class=kneedWalker, transcription=dm.GaussLobatto(num_segments=10, order=3), ode_init_kwargs={'states_ref': midphase_guess}))

    threelink.set_time_options(fix_initial=True, fix_duration=True, duration_val=duration_threelink, duration_ref=duration_threelink, units='s')

    threelink.add_parameter('L', val=L, units='m', static_target=True)
    threelink.add_parameter('a1', val=a1, units='m', static_target=True)
    threelink.add_parameter('a2', val=a2, units='m', static_target=True)
    threelink.add_parameter('b1', val=b1, units='m', static_target=True)
    threelink.add_parameter('b2', val=b2, units='m', static_target=True)
    threelink.add_parameter('m_H', val=m_H, units='kg', static_target=True)
    threelink.add_parameter('m_t', val=m_t, units='kg', static_target=True)
    threelink.add_parameter('m_s', val=m_s, units='kg', static_target=True)

    threelink.add_state('q1', fix_initial=True, lower = -4, upper = 4, rate_source='q1_dot', units='rad')
    threelink.add_state('q1_dot', fix_initial=True, lower=-20, upper = 20, rate_source='q1_dotdot_3', units='rad/s')
    threelink.add_state('q2', fix_initial=True, lower = -4, upper = 4, rate_source='q2_dot', units='rad')
    threelink.add_state('q2_dot', fix_initial=True, lower = -20, upper = 20,  rate_source='q2_dotdot_3', units='rad/s')
    threelink.add_state('q3', fix_initial=True, lower = -4, upper= 4, rate_source='q3_dot', units='rad')
    threelink.add_state('q3_dot', fix_initial=True, rate_source='q3_dotdot_3', units='rad/s')
    threelink.add_state('cost', fix_initial=True, rate_source='costrate')

    threelink.add_control('tau', fix_initial=False, units='N*m')

    # initial contraints
    threelink.add_boundary_constraint('q1', loc='initial', equals=states_init['q1'], units='rad')
    threelink.add_boundary_constraint('q1_dot', loc='initial', equals=states_init['q1_dot'], units='rad/s')
    threelink.add_boundary_constraint('q2', loc='initial', equals=states_init['q2'], units='rad')
    threelink.add_boundary_constraint('q2_dot', loc='initial', equals=states_init['q2_dot'], units='rad/s')

    # end constraints  q3 = q2 at end of three link phase
    #threelink.add_boundary_constraint('q3', loc='final', equals='q2', units='rad')

    #threelink.add_objective('cost', loc='final')

    #####

    lockphase = traj.add_phase('lockphase', dm.Phase(ode_class=kneedWalker, transcription=dm.GaussLobatto(num_segments=10, order=3), ode_init_kwargs={'states_ref': states_final}))

    lockphase.set_time_options(fix_initial=True, initial_val=12, fix_duration=True, duration_val=duration_lockphase, duration_ref=duration_lockphase, units='s') # set time of simulation    

    #states
    lockphase.add_state('q1', fix_initial=True, lower = -4, upper = 4, rate_source='q1_dot', units='rad')
    lockphase.add_state('q1_dot', fix_initial=True, lower=-20, upper = 20, rate_source='q1_dotdot', units='rad/s')
    lockphase.add_state('q2', fix_initial=True, lower = -4, upper = 4, rate_source='q2_dot', units='rad')
    lockphase.add_state('q2_dot', fix_initial=True, lower = -20, upper = 20,  rate_source='q2_dotdot', units='rad/s')
    lockphase.add_state('cost', fix_initial=True, rate_source='costrate',)

    lockphase.add_control('tau', fix_initial=False, units='N*m') # add control torque

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
    lockphase.add_boundary_constraint('q1', loc='final', equals=states_final['q1'], units='rad')
    lockphase.add_boundary_constraint('q1_dot', loc='final', equals=states_final['q1_dot'], units='rad/s')
    lockphase.add_boundary_constraint('q2', loc='final', equals=states_final['q2'], units='rad')
    lockphase.add_boundary_constraint('q2_dot', loc='final', equals=states_final['q2_dot'], units='rad/s')

    # add objective - TO DO - add cost function and obj
    #lockphase.add_objective('cost', loc='final')

    threelink.add_objective('cost', loc='final')

    p.setup(check=True)

    # initial guess / time - lockphase
    #p.set_val('traj.lockphase.states:q1', lockphase.interp(ys=['traj.threelink.states:q1', states_final['q1']], nodes='state_input'), units='rad')
    #p.set_val('traj.lockphase.states:q1_dot', lockphase.interp(ys=['traj.threelink.states:q1_dot', states_final['q1_dot']], nodes='state_input'), units='rad/s')
    #p.set_val('traj.lockphase.states:q2', lockphase.interp(ys=['traj.threelink.states:q2', states_final['q2']], nodes='state_input'), units='rad')
    #p.set_val('traj.lockphase.states:q2_dot', lockphase.interp(ys=['traj.threelink.states:q2_dot', states_final['q2_dot']], nodes='state_input'), units='rad/s')
    #p.set_val('traj.lockphase.states:cost', lockphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 500, 1000], nodes='state_input'))
    #p.set_val('traj.lockphase.controls:tau', lockphase.interp(ys=[0, 10], nodes='control_input'), units='N*m')



    # initial guess / time - three link
    p.set_val('traj.threelink.t_initial', 0.0)
    p.set_val('traj.threelink.states:q1', threelink.interp(ys=[states_init['q1'], midphase_guess['q1']], nodes='state_input'), units='rad')
    p.set_val('traj.threelink.states:q1_dot', threelink.interp(ys=[states_init['q1_dot'], midphase_guess['q1_dot']], nodes='state_input'), units='rad/s')
    p.set_val('traj.threelink.states:q2', threelink.interp(ys=[states_init['q2'], midphase_guess['q2']], nodes='state_input'), units='rad')
    p.set_val('traj.threelink.states:q2_dot', threelink.interp(ys=[states_init['q2_dot'], midphase_guess['q2_dot']], nodes='state_input'), units='rad/s')
    p.set_val('traj.threelink.states:q3', threelink.interp(ys=[states_init['q3'], midphase_guess['q3']], nodes='state_input'), units='rad')
    p.set_val('traj.threelink.states:q3_dot', threelink.interp(ys=[states_init['q3_dot'], midphase_guess['q3_dot']], nodes='state_input'), units='rad/s')
    p.set_val('traj.threelink.states:cost', threelink.interp(xs=[0, 10, duration_threelink], ys=[0, 500, 1000], nodes='state_input'))
    p.set_val('traj.threelink.controls:tau', threelink.interp(ys=[0, 10], nodes='control_input'), units='N*m')


    #traj.add_linkage_constraint('threelink', 'lockphase', 'q1', 'q1')
    #traj.add_linkage_constraint('threelink', 'lockphase', 'q1_dot', 'q1_dot')
    #traj.add_linkage_constraint('threelink', 'lockphase', 'q2', 'q2')
    #traj.add_linkage_constraint('threelink', 'lockphase', 'q2_dot', 'q2_dot')
    #traj.add_linkage_constraint('threelink', 'lockphase', 'cost', 'cost')
    #traj.add_linkage_constraint('threelink', 'lockphase', 'tau', 'tau')
    traj.link_phases(['threelink', 'lockphase'],connected=True)# vars=['q1', 'q2', 'q1_dot', 'q2_dot', 'tau', 'cost'])
    # need to add other phases


    #.model.traj.phases.lockphase.set_refine_options(refine=True, tol=1.0E-6)
    #p.model.traj.phases.threelink.set_refine_options(refine=True, tol=1.0E-6)


    # simulate and run problem
    dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10}, make_plots=True)

    om.n2(p)

    # print values - since there is no objective atm this doesnt mean anything
    #print('L:', p.get_val('L', units='m'))
    # print('q1:', p.get_val('q1', units='rad'))
    # print('q1_dot:', p.get_val('q1_dot', units='rad/s'))
    # print('q2:', p.get_val('q2', units='rad'))
    # print('q2_dot:', p.get_val('q2_dot', units='rad/s'))
    cost = p.get_val('traj.threelink.states:cost')[-1]
    print('cost: ', cost)

    sim_sol = om.CaseReader('dymos_simulation.db').get_case('final')

    # plot time history

    plot_results([('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:q1','time', 'q1'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:q2','time','q2'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:q1_dot','time','q1_dot'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:q2_dot','time','q2_dot'),
                  ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.controls:tau','time','tau'),
                  ('traj.lockphase.timeseries.time', 'traj.lockphase.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p,p_sim=sim_sol)
    plt.savefig('openloop_locked_kneedwalker.pdf', bbox_inches='tight')


    plot_results([('traj.threelink.timeseries.time','traj.threelink.timeseries.states:q1','time', 'q1'),
                  ('traj.threelink.timeseries.time','traj.threelink.timeseries.states:q2','time','q2'),
                  ('traj.threelink.timeseries.time', 'traj.threelink.timeseries.states:q3', 'time', 'q3'),
                  ('traj.threelink.timeseries.time','traj.threelink.timeseries.states:q1_dot','time','q1_dot'),
                  ('traj.threelink.timeseries.time','traj.threelink.timeseries.states:q2_dot','time','q2_dot'),
                  ('traj.threelink.timeseries.time','traj.threelink.timeseries.controls:tau','time','tau'),
                  ('traj.threelink.timeseries.time', 'traj.threelink.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p,p_sim=sim_sol)
    plt.savefig('openloop_three_kneedwalker.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()

