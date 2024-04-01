import numpy as np
import openmdao.api as om
import dymos as dm
from compass_notorque import system_passive
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results
import numpy as np


def main(co_design=True):
    duration_lockphase = 1 # duration of locked knee phase

    # defining paramters of the legged walker

    a = 0.5
    b = 0.5
    body_mass = 6
    carrier_mass = 3
    mh = 10
    m = 5
    l = a + b
    phi = 0.05 # slope - 3 degrees
    phi_contraint = -2*phi

    density = 5

    """ 
    walker will complete one full cycle -- states will be the same at the end as they were at the beginning
    """
    states_init = {'x1': -0.3, 'x3': -0.41215, 'x2': 0.2038, 'x4': -1.0501} # initial conditions
    states_final = {'x1': 0.0264, 'x3': -0.895, 'x2': -0.1264, 'x4': -0.669} # final guess
    #states_final = {'x1': .1, 'x3': 0, 'x2': -0.2, 'x4': 0}

    p = om.Problem()

    p.driver = om.pyOptSparseDriver() # set driver to ipopt
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False
    p.driver.declare_coloring(orders=None)
    p.driver.opt_settings['max_iter'] = 300
    p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['print_level'] = 5
    # prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['nlp_scaling_method'] = 'none'
    p.driver.opt_settings['tol'] = 1.0e-10
    p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    p.driver.opt_settings['limited_memory_max_history'] = 100
    p.driver.opt_settings['limited_memory_max_skipping'] = 5
    ### prob.driver.opt_settings['mu_init'] = 1e-1  # starting from small mu_init might help when you have a good guess
    p.driver.opt_settings['mu_strategy'] = 'monotone'  # 'monotone' is usually more robust, `adaptive` would be more efficient
    p.driver.opt_settings['mu_min'] = 1e-8   # only for adaptive
    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
    # restortion phase
    p.driver.opt_settings['required_infeasibility_reduction'] = 0.99
    p.driver.opt_settings['max_resto_iter'] = 100

    p2 = om.Problem()

    p2.driver = om.pyOptSparseDriver() # set driver to ipopt
    p2.driver.options['optimizer'] = 'IPOPT'
    p2.driver.options['print_results'] = False
    p2.driver.declare_coloring(orders=None)
    p2.driver.opt_settings['max_iter'] = 300
    p2.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p2.driver.opt_settings['print_level'] = 5
    # prob.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p2.driver.opt_settings['nlp_scaling_method'] = 'none'
    p2.driver.opt_settings['tol'] = 1.0e-10
    p2.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    p2.driver.opt_settings['limited_memory_max_history'] = 100
    p2.driver.opt_settings['limited_memory_max_skipping'] = 5
    ### prob.driver.opt_settings['mu_init'] = 1e-1  # starting from small mu_init might help when you have a good guess
    p2.driver.opt_settings['mu_strategy'] = 'monotone'  # 'monotone' is usually more robust, `adaptive` would be more efficient
    p2.driver.opt_settings['mu_min'] = 1e-8   # only for adaptive
    p2.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
    # restortion phase
    p2.driver.opt_settings['required_infeasibility_reduction'] = 0.99
    p2.driver.opt_settings['max_resto_iter'] = 100

    
    traj = p.model.add_subsystem('traj', dm.Trajectory()) # init trajectory

    #add init phase
    initphase = traj.add_phase('initphase', dm.Phase(ode_class=system_passive, transcription=dm.GaussLobatto(num_segments=25, order=3), ode_init_kwargs={'states_ref': states_final}))
    initphase.set_time_options(fix_initial=True, initial_val=0, fix_duration=False, units='s') # set time options for simulation

    #states for init phase
    initphase.add_state('x1', upper= 1, lower= -1, fix_initial=True, rate_source='x1_dot', units='rad')
    initphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
    initphase.add_state('x2', upper= 1, lower= -1, fix_initial=True, rate_source='x2_dot', units='rad')
    initphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
    initphase.add_state('cost', fix_initial=True, rate_source='costrate')

    initphase.add_control('tau', lower = -10, upper = 10, fix_initial=False, units='N*m') # add control torque

    # add initial conditions for init phase
    initphase.add_boundary_constraint('x1', loc='initial', equals=states_init['x1'])
    initphase.add_boundary_constraint('x2', loc='initial', equals=states_init['x2'])
    initphase.add_boundary_constraint('x3', loc='initial', equals=states_init['x3'])
    initphase.add_boundary_constraint('x4', loc='initial', equals=states_init['x4'])

    # paramaters - same for both phases
    initphase.add_parameter('a', val=a, units='m', static_target=True)
    initphase.add_parameter('b', val=b, units='m', static_target=True)
    initphase.add_parameter('mh', val=mh, units='kg', static_target=True)
    initphase.add_parameter('m', val=m, units='kg', static_target=True)

    # transition boudary contraints
    initphase.add_boundary_constraint('phi_bounds', loc='final', equals=phi_contraint,  units='rad')

    initphase.add_objective('cost')

    p.setup(check=True)

    p.set_val('traj.initphase.states:x1', initphase.interp(ys=[states_init['x1'], states_final['x1']], nodes='state_input'), units='rad')
    p.set_val('traj.initphase.states:x3', initphase.interp(ys=[states_init['x3'], states_final['x3']], nodes='state_input'), units='rad/s')
    p.set_val('traj.initphase.states:x2', initphase.interp(ys=[states_init['x2'], states_final['x2']], nodes='state_input'), units='rad')
    p.set_val('traj.initphase.states:x4', initphase.interp(ys=[states_init['x4'], states_final['x4']], nodes='state_input'), units='rad/s')
    p.set_val('traj.initphase.states:cost', initphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
    p.set_val('traj.initphase.controls:tau', initphase.interp(ys=[0, 10], nodes='control_input'), units='N*m') 


    dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 50})

    #om.n2(p)

    # print cost
    cost = p.get_val('traj.initphase.states:cost')[-1]
    print('cost: ', cost)



    sim_sol = om.CaseReader('dymos_simulation.db').get_case('final')

    # plot time history

    plot_results([('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x1','time', 'q1'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x2','time','q2'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x3','time','q1_dot'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x4','time','q2_dot'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.controls:tau','time','tau'),
                  ('traj.initphase.timeseries.time', 'traj.initphase.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p,p_sim=sim_sol)
    plt.savefig('compass_passive_noopt.pdf', bbox_inches='tight')

    # add end conditions to be used with control
    x1_end = p.get_val('traj.initphase.states:x1')[-1]
    x2_end = p.get_val('traj.initphase.states:x2')[-1]
    x3_end = p.get_val('traj.initphase.states:x3')[-1]
    x4_end = p.get_val('traj.initphase.states:x4')[-1]
    x1_end = x1_end[0]
    x2_end = x2_end[0]
    x3_end = x3_end[0]
    x4_end = x4_end[0]

    x1_initphase = p.get_val('traj.initphase.states:x1')
    x2_initphase = p.get_val('traj.initphase.states:x2')
    x3_initphase = p.get_val('traj.initphase.states:x3')
    x4_initphase = p.get_val('traj.initphase.states:x4')

    #position = np.concatenate((x1_initphase, x2_lockphase, x1_lockphase2))
    fig, ax = plt.subplots()

    pos = x1_initphase
    velo = x3_initphase

    ax.plot(x1_initphase, x3_initphase, linewidth=1.0, label='swing foot')
    ax.plot(x2_initphase, x4_initphase, linewidth=1.0, label='stance foot')

    #ax.plot(x1_lockphase, x3_lockphase, linewidth=1.0, label='lockphase x1')
    #ax.plot(x2_lockphase, x4_lockphase, linewidth=1.0, label='lockphase x2')

    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    #ax.legend()

    plt.savefig('limitcycle_opt=false.pdf')

    ## plot motion

    # get states
    x1arr = x1_initphase
    x2arr = x2_initphase

    num_points = len(x1arr)

    #plot animation
    from animate import animate_compass
    animate_compass(x1arr.reshape(num_points), x2arr.reshape(num_points), a, b, phi, 'opt_false.gif', saveFig=True)


    """
    run problem with control / design vars
    """
    from compass_torque import system_active

    states_init = {'x1': -0.3, 'x3': -0.41215, 'x2': 0.2038, 'x4': -1.0501} # initial conditions
    states_final = {'x1': x1_end, 'x3': x3_end, 'x2': x2_end, 'x4': x4_end} # final guess


    design_vars = p2.model.add_subsystem('design_vars', om.IndepVarComp(), promotes_outputs=['*'])
    design_vars.add_output('a', val=a, units='m')
    design_vars.add_output('b', val=b, units='m')
    design_vars.add_output('density', val=density, units='kg/m')
 
    if co_design:
        # add design var a 
        p2.model.add_design_var('a', units='m', lower=0.1, upper=1, ref=0.5)
        p2.model.add_design_var('b', units='m', lower=0.1, upper = 1, ref=0.5)

        leg_mass_comp = om.ExecComp('m=density*(a+b)')
        leg_mass = p2.model.add_subsystem('m', leg_mass_comp, promotes=['*'])


    traj = p2.model.add_subsystem('traj', dm.Trajectory()) # init trajectory

    #add init phase
    initphase = traj.add_phase('initphase', dm.Phase(ode_class=system_passive, transcription=dm.GaussLobatto(num_segments=25, order=3), ode_init_kwargs={'states_ref': states_final}))
    initphase.set_time_options(fix_initial=True, initial_val=0, fix_duration=False, units='s') # set time options for simulation

    #states for init phase
    initphase.add_state('x1', upper= 1, lower= -1, fix_initial=True, fix_final=True, rate_source='x1_dot', units='rad')
    initphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
    initphase.add_state('x2', upper= 1, lower= -1, fix_initial=True, fix_final=True, rate_source='x2_dot', units='rad')
    initphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
    initphase.add_state('cost', fix_initial=True, rate_source='costrate')

    initphase.add_control('tau', lower = -10, upper = 10, fix_initial=False, units='N*m') # add control torque

    # add initial conditions for init phase
    initphase.add_boundary_constraint('x1', loc='initial', equals=states_init['x1'])
    initphase.add_boundary_constraint('x2', loc='initial', equals=states_init['x2'])
    initphase.add_boundary_constraint('x3', loc='initial', equals=states_init['x3'])
    initphase.add_boundary_constraint('x4', loc='initial', equals=states_init['x4'])

    #add end constraints based off of previous sim
    initphase.add_boundary_constraint('x1', loc='final', equals=states_final['x1'])
    initphase.add_boundary_constraint('x2', loc='final', equals=states_final['x2'])

    # paramaters - same for both phases
    initphase.add_parameter('a', val=a, units='m', static_target=True)
    initphase.add_parameter('b', val=b, units='m', static_target=True)
    initphase.add_parameter('mh', opt=True, val=mh, lower=5, upper=1000, units='kg', static_target=True)
    initphase.add_parameter('m', val=m, units='kg', static_target=True)


    # set design var a to be a and b
    p2.model.connect('a', 'traj.initphase.parameters:a')
    p2.model.connect('b', 'traj.initphase.parameters:b')
    p2.model.connect('m', 'traj.initphase.parameters:m')

    #initphase.add_timeseries_output('costrate', output_name='costrate')

    initphase.add_objective('mh', scaler=-1)

    p2.setup(check=True)

    p2.set_val('traj.initphase.states:x1', initphase.interp(ys=[states_init['x1'], states_final['x1']], nodes='state_input'), units='rad')
    p2.set_val('traj.initphase.states:x3', initphase.interp(ys=[states_init['x3'], states_final['x3']], nodes='state_input'), units='rad/s')
    p2.set_val('traj.initphase.states:x2', initphase.interp(ys=[states_init['x2'], states_final['x2']], nodes='state_input'), units='rad')
    p2.set_val('traj.initphase.states:x4', initphase.interp(ys=[states_init['x4'], states_final['x4']], nodes='state_input'), units='rad/s')
    p2.set_val('traj.initphase.states:cost', initphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
    p2.set_val('traj.initphase.controls:tau', initphase.interp(ys=[0, 10], nodes='control_input'), units='N*m') 

    #traj.simulate(times_per_seg=20, method='Radau')
    #print(p.get_val('traj.initphase.states:x1'))


    dm.run_problem(p2, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 50})

    #om.n2(p)

    # print cost
    #cost = p2.get_val('traj.initphase.timeseries.costrate')[1]
    #print('cost: ', cost)
    print('a: ', p2.get_val('a', units='m'))
    print('b: ', p2.get_val('b', units='m'))
    print('leg mass:', p2.get_val('traj.initphase.parameters:m', units='kg'))
    #print('body mass: ', p.get_val('mass_body', units='kg'))
    #print('carrier mass: ', p.get_val('mass_carrier', units='kg'))
    #print('carried mass: ', p.get_val('mass_carrying', units='kg'))
    print('hip mass: ', p2.get_val('traj.initphase.parameters:mh', units='kg'))

    a = p2.get_val('a', units='m')
    b = p2.get_val('b', units='m')

    sim_sol = om.CaseReader('dymos_simulation.db').get_case('final')

    # plot time history

    plot_results([('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x1','time', 'q1'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x2','time','q2'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x3','time','q1_dot'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.states:x4','time','q2_dot'),
                  ('traj.initphase.timeseries.time','traj.initphase.timeseries.controls:tau','time','tau'),
                  ('traj.initphase.timeseries.time', 'traj.initphase.timeseries.states:cost', 'time', 'cost')],
                  title='Time History',p_sol=p2,p_sim=sim_sol)
    plt.savefig('compassgait_active_opt=true.pdf', bbox_inches='tight')


    x1_initphase = p2.get_val('traj.initphase.states:x1')
    x2_initphase = p2.get_val('traj.initphase.states:x2')
    x3_initphase = p2.get_val('traj.initphase.states:x3')
    x4_initphase = p2.get_val('traj.initphase.states:x4')

    #position = np.concatenate((x1_initphase, x2_lockphase, x1_lockphase2))
    fig, ax = plt.subplots()

    pos = x1_initphase
    velo = x3_initphase

    ax.plot(x1_initphase, x3_initphase, linewidth=1.0, label='swing foot')
    ax.plot(x2_initphase, x4_initphase, linewidth=1.0, label='stance foot')

    #ax.plot(x1_lockphase, x3_lockphase, linewidth=1.0, label='lockphase x1')
    #ax.plot(x2_lockphase, x4_lockphase, linewidth=1.0, label='lockphase x2')

    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    #ax.legend()

    plt.savefig('limitcycle_opt=true.pdf')

    ## plot motion

    # get states
    x1arr = x1_initphase
    x2arr = x2_initphase

    num_points = len(x1arr)

    #plot animation
    from animate import animate_compass
    animate_compass(x1arr.reshape(num_points), x2arr.reshape(num_points), a, b, phi, 'opt_true.gif', saveFig=True)



if __name__ == '__main__':
    main()
