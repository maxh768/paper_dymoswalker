import numpy as np
import openmdao.api as om
import dymos as dm
from compass_notorque import system_passive
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results
import numpy as np
from numpy.linalg import inv

def main():
    duration_lockphase = 1 # duration of locked knee phase

    # defining paramters of the legged walker

    a = 0.5
    b = 0.5
    mh = 10
    m = 5
    l = a + b
    phi = 0.05 # slope - 3 degrees
    phi_contraint = -2*phi

    """
    main for loop - will run through one gait (until swing leg hits ground) then will 
    take final conditions and use those as initial conditions for the next gait
    """

    #original initial conditions
    states_init = {'x1': -0.3, 'x3': -0.41215, 'x2': 0.2038, 'x4': -1.0501} # initial conditions
    states_final = {'x1': 0.02, 'x3': -0.9, 'x2': -0.1, 'x4': -0.7} # final guess

    # number of iterations
    iterations = 10

    for i in range(iterations):

        if (i!=0):
            states_init = {'x1': states_init[0], 'x2': states_init[1], 'x3': states_init[2], 'x4': states_init[3]}

        states_ref = states_init

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

        traj = p.model.add_subsystem('traj', dm.Trajectory()) # init trajectory

        lockphase = traj.add_phase('lockphase', dm.Phase(ode_class=system_passive, transcription=dm.GaussLobatto(num_segments=20, order=3), ode_init_kwargs={'states_ref': states_ref}))
        lockphase.set_time_options(fix_initial=True, fix_duration=False, initial_val=0, units='s') # set time options for simulation

        #states for lockphase 1 phase
        lockphase.add_state('x1', fix_initial=True, rate_source='x1_dot', units='rad')
        lockphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
        lockphase.add_state('x2', fix_initial=True, rate_source='x2_dot', units='rad')
        lockphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
        lockphase.add_state('cost', fix_initial=True, rate_source='costrate')

        lockphase.add_control('tau', lower = -10, upper = 10, fix_initial=False, units='N*m') # add control torque

        # set initial conditions
        lockphase.add_boundary_constraint('x1', loc='initial', equals=states_init['x1'])
        lockphase.add_boundary_constraint('x2', loc='initial', equals=states_init['x2'])
        lockphase.add_boundary_constraint('x3', loc='initial', equals=states_init['x3'])
        lockphase.add_boundary_constraint('x4', loc='initial', equals=states_init['x4'])

        # add paramaters
        lockphase.add_parameter('a', val=a, units='m', static_target=True)
        lockphase.add_parameter('b', val=b, units='m', static_target=True)
        lockphase.add_parameter('mh', val=mh, opt=False, units='kg', static_target=True)
        lockphase.add_parameter('m', val=m, units='kg', static_target=True)

        # force legs to be at given angle at start and end
        lockphase.add_boundary_constraint('phi_bounds', loc='final', equals=phi_contraint,  units='rad')
        #lockphase.add_boundary_constraint('phi_bounds', loc='initial', equals=phi_contraint,  units='rad')

        # add dummy objective for now
        lockphase.add_objective('mh')

        p.setup(check=True)

        #interpolate values onto traj
        p.set_val('traj.lockphase.states:x1', lockphase.interp(ys=[states_init['x1'], states_final['x1']], nodes='state_input'), units='rad')
        p.set_val('traj.lockphase.states:x3', lockphase.interp(ys=[states_init['x3'], states_final['x3']], nodes='state_input'), units='rad/s')
        p.set_val('traj.lockphase.states:x2', lockphase.interp(ys=[states_init['x2'], states_final['x2']], nodes='state_input'), units='rad')
        p.set_val('traj.lockphase.states:x4', lockphase.interp(ys=[states_init['x4'], states_final['x4']], nodes='state_input'), units='rad/s')
        p.set_val('traj.lockphase.states:cost', lockphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
        p.set_val('traj.lockphase.controls:tau', lockphase.interp(ys=[0, 10], nodes='control_input'), units='N*m') 

        #run problem
        dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10})

        sim_sol = om.CaseReader('dymos_simulation.db').get_case('final') # simulation solution

        #plot results from each phase - will get erased each time loop runs
        plot_results([('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x1','time', 'q1'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x2','time','q2'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x3','time','q1_dot'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x4','time','q2_dot')],
                    title='Time History',p_sol=p,p_sim=sim_sol)
        plt.savefig('compassgait_lockhpase.pdf', bbox_inches='tight')

        # get final values from traj
        x1_end = p.get_val('traj.lockphase.states:x1')[-1]
        x2_end = p.get_val('traj.lockphase.states:x2')[-1]
        x3_end = p.get_val('traj.lockphase.states:x3')[-1]
        x4_end = p.get_val('traj.lockphase.states:x4')[-1]
        x3_end = x3_end[0]
        x4_end = x4_end[0]


        """
        transition equations
        """

        # calculate alpha at end of gait
        alpha = np.abs((x2_end[0] - x1_end[0])) / 2

        # Q+ matrix
        Qp11 = m*b*(b-l*np.cos(2*alpha))
        Qp12 = m*l*(l-b*np.cos(2*alpha)) + m*a**2 + mh*l**2
        Qp21 = m*b**2
        Qp22 = -m*b*l*np.cos(2*alpha)

        Qm11 = -m*a*b
        Qm12 = -m*a*b + (mh*l**2 + 2*m*a*l)*np.cos(2*alpha)
        Qm21 = 0
        Qm22 = -m*a*b

        Qplus = np.array([[Qp11, Qp12], [Qp21, Qp22]])
        Qminus = np.array([[Qm11, Qm12], [Qm21, Qm22]])

        Qplus_inverted = inv(Qplus)

        H = np.dot(Qplus_inverted, Qminus)


        newx3 = H[0,0]*x3_end + H[0, 1]*x4_end
        newx4 = H[1,0]*x3_end + H[1, 1]*x4_end

        states_init = [x2_end[0], x1_end[0], newx3, newx4]

        # for plotting

        if (i==0):
            x1arr = p.get_val('traj.lockphase.timeseries.states:x1')
            x2arr = p.get_val('traj.lockphase.timeseries.states:x2')
            x3arr = p.get_val('traj.lockphase.timeseries.states:x3')
            x4arr = p.get_val('traj.lockphase.timeseries.states:x4')
            num_iter = len(x1arr)
            timearr = p.get_val('traj.lockphase.timeseries.time')
            endtimes = timearr[-1]
        else:
            x1arr = np.concatenate((x1arr, p.get_val('traj.lockphase.timeseries.states:x1')))
            x2arr = np.concatenate((x2arr, p.get_val('traj.lockphase.timeseries.states:x2')))
            x3arr = np.concatenate((x3arr, p.get_val('traj.lockphase.timeseries.states:x3')))
            x4arr = np.concatenate((x4arr, p.get_val('traj.lockphase.timeseries.states:x4')))
            intertime = p.get_val('traj.lockphase.timeseries.time') + timearr[-1]
            timearr = np.concatenate((timearr, intertime))
            endtimes = np.concatenate((endtimes, p.get_val('traj.lockphase.timeseries.time')[-1]))

    #end for loop

    #plot limit cycle
    fig, ax = plt.subplots()


    ax.plot(x1arr, x3arr, linewidth=1.0, label='swing foot')
    ax.plot(x2arr, x4arr, linewidth=1.0, label='stance foot')
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    ax.legend()
    plt.savefig('limitcycle_compass.pdf')
    

    # animate motion
    from animate import animate_compass
    num_points = len(x1arr)
    animate_compass(x1arr.reshape(num_points), x2arr.reshape(num_points), a, b, phi, saveFig=True, name='runmultpassive.gif',gif_fps=40, iter=iterations, num_iter_points=num_iter)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('States Over Entire Range - Passive System')
    fig.tight_layout()
    ax1.plot(timearr, x1arr)
    ax2.plot(timearr, x2arr)
    ax3.plot(timearr, x3arr)
    ax4.plot(timearr, x4arr)

    ax1.set_ylabel('x1 (rad)')
    ax2.set_ylabel('x2 (rad)')
    ax3.set_ylabel('x3 (rad/s)')
    ax4.set_ylabel('x4 (rad/s)')
    ax4.set_xlabel('time (s)')

    plt.savefig('total_timehistory', bbox_inches='tight')
    print("Final Time = ", timearr[-1])
    avgcyc = np.mean(endtimes)
    print('Average Time/Cycle = ', avgcyc)



if __name__ == '__main__':
    main()

        




