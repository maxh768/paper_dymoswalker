import numpy as np
import openmdao.api as om
import dymos as dm
from compassdynamics import system
import matplotlib.pyplot as plt
from dymos.examples.plotting import plot_results
import numpy as np

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
    states_init = [0, 0, 1, -0.2]

    #create plotting arrays
    x1series = [0, 0, 0]
    x2series = [0, 0, 0]
    x3series = [0, 0, 0]
    x4series = [0, 0, 0]

    for i in range(3):

        
        states_ref = {'x1': states_init[0], 'x3': states_init[1], 'x2': states_init[2], 'x4': states_init[3]}

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

        lockphase = traj.add_phase('lockphase', dm.Phase(ode_class=system, transcription=dm.GaussLobatto(num_segments=20, order=3), ode_init_kwargs={'states_ref': states_ref}))
        lockphase.set_time_options(fix_initial=True, fix_duration=False, initial_val=0, units='s') # set time options for simulation

        #states for lockphase 1 phase
        lockphase.add_state('x1', fix_initial=True, rate_source='x1_dot', units='rad')
        lockphase.add_state('x3', fix_initial=True, rate_source='x3_dot', units='rad/s')
        lockphase.add_state('x2', fix_initial=True, rate_source='x2_dot', units='rad')
        lockphase.add_state('x4', fix_initial=True,  rate_source='x4_dot', units='rad/s')
        lockphase.add_state('cost', fix_initial=True, rate_source='costrate')

        lockphase.add_control('tau', lower = -10, upper = 10, fix_initial=False, units='N*m') # add control torque

        # set initial conditions
        lockphase.add_boundary_constraint('x1', loc='initial', equals=states_init[0])
        lockphase.add_boundary_constraint('x2', loc='initial', equals=states_init[1])
        lockphase.add_boundary_constraint('x3', loc='initial', equals=states_init[2])
        lockphase.add_boundary_constraint('x4', loc='initial', equals=states_init[3])

        # add paramaters
        lockphase.add_parameter('a', val=a, units='m', static_target=True)
        lockphase.add_parameter('b', val=b, units='m', static_target=True)
        lockphase.add_parameter('mh', val=mh, units='kg', static_target=True)
        lockphase.add_parameter('m', val=m, units='kg', static_target=True)

        # force legs to be at given angle at start and end
        lockphase.add_boundary_constraint('phi_bounds', loc='final', equals=phi_contraint,  units='rad')
        #lockphase.add_boundary_constraint('phi_bounds', loc='initial', equals=phi_contraint,  units='rad')

        # add dummy objective for now
        lockphase.add_objective('cost')

        p.setup(check=True)

        #interpolate values onto traj
        p.set_val('traj.lockphase.states:x1', lockphase.interp(ys=[states_init[0], 0], nodes='state_input'), units='rad')
        p.set_val('traj.lockphase.states:x3', lockphase.interp(ys=[states_init[1], 0], nodes='state_input'), units='rad/s')
        p.set_val('traj.lockphase.states:x2', lockphase.interp(ys=[states_init[2], 0], nodes='state_input'), units='rad')
        p.set_val('traj.lockphase.states:x4', lockphase.interp(ys=[states_init[3], 0], nodes='state_input'), units='rad/s')
        p.set_val('traj.lockphase.states:cost', lockphase.interp(xs=[0, 2, duration_lockphase], ys=[0, 50, 100], nodes='state_input'))
        p.set_val('traj.lockphase.controls:tau', lockphase.interp(ys=[0, 10], nodes='control_input'), units='N*m') 

        #run problem
        dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10})

        sim_sol = om.CaseReader('dymos_simulation.db').get_case('final') # simulation solution

        #plot results from each phase - will get erased each time loop runs
        plot_results([('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x1','time', 'q1'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x2','time','q2'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x3','time','q1_dot'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.states:x4','time','q2_dot'),
                    ('traj.lockphase.timeseries.time','traj.lockphase.timeseries.controls:tau','time','tau'),
                    ('traj.lockphase.timeseries.time', 'traj.lockphase.timeseries.states:cost', 'time', 'cost')],
                    title='Time History',p_sol=p,p_sim=sim_sol)
        plt.savefig('compassgait_lockhpase.pdf', bbox_inches='tight')

        # get final values from traj
        x1_end = p.get_val('traj.lockphase.states:x1')[-1]
        x2_end = p.get_val('traj.lockphase.states:x2')[-1]
        x3_end = p.get_val('traj.lockphase.states:x3')[-1]
        x4_end = p.get_val('traj.lockphase.states:x4')[-1]

        #get timeseries for animation and plotting
        x1series[i] = p.get_val('traj.lockphase.states:x1')
        x2series[i] = p.get_val('traj.lockphase.states:x2')
        x3series[i] = p.get_val('traj.lockphase.states:x3')
        x4series[i] = p.get_val('traj.lockphase.states:x4')

        print('iteration: ', i)

        """
        transition equations
        """

        # calculate alpha at end of gait
        alpha = np.abs((x1_end - x2_end)) / 2

        # calculating transition matrices for phase change
        Q11_m = -m*a*b; Q12_m = -m*a*b + ((mh*l**2) + 2*m*a*l)*np.cos(2*alpha); Q22_m = Q11_m
        Q11_p = m*b*(b - l*np.cos(2*alpha)); Q12_p = m*l*(l-b*np.cos(2*alpha)) + m*a**2 + mh*l**2; Q21_p = m*b**2; Q22_p = -m*b*l*np.cos(2*alpha)

        kq_p = 1 / (Q11_p*Q22_p - Q12_p*Q21_p) # inverse constant

        # transition matrix for velocities
        P11 = kq_p*(Q22_p*Q11_m ); P12 = kq_p*(Q22_p*Q12_m - Q12_p*Q22_m); P21 = kq_p*(-Q21_p*Q11_m); P22 = kq_p*(-Q21_p*Q12_m + Q11_p*Q22_m)

        newx3 = (P11*x3_end[0] + P12*x4_end[0])
        newx4 = (P21*x3_end[0] + P22*x4_end[0])

        print('type ', type(x1_end))

        states_init = [x2_end[0], x1_end[0], newx3[0], newx4[0]]
        

    #end for loop

    #plot limit cycle
    fig, ax = plt.subplots()

    pos = np.concatenate((x1series[0], x2series[1], x1series[2]))
    velo = np.concatenate((x3series[0], x4series[1], x3series[2]))

    ax.plot(pos, velo, linewidth=1.0, label='swing foot')
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    plt.savefig('limitcycle_compass.pdf')

    # animate motion
    x1arr = np.concatenate((x1series[0], x1series[1], x1series[2]))
    x2arr = np.concatenate((x2series[0], x2series[1], x2series[2]))
    num_points = len(x1arr)
    from animate import animate_compass
    animate_compass(x1arr.reshape(num_points), x2arr.reshape(num_points), a, b, phi, saveFig=True)



if __name__ == '__main__':
    main()

        




