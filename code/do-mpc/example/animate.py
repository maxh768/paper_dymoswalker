import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation


def animate_compass(x1, x2, a, b, phi, name='compass.gif', interval = 20, saveFig=False, gif_fps=20, iter=1, num_iter_points=90, time=2.019):
    #x1: back leg
    #x2: front leg
    #phi: angle of slope
    l = a + b

    # path of stance leg tip (hip)
    x_hip = -l*np.sin(x2)
    y_hip = l*np.cos(x2)

    # path of the swing leg foot
    #intermediate x and y vals
    x_hip2swing = l*np.sin(x1) # changes sign
    y_hip2swing = -l*np.cos(x1) # always negative

    # acutal x and y vals of swing foot
    x_swing = x_hip+x_hip2swing # x and y coords of swing foot wrt stance foot
    y_swing = y_hip+y_hip2swing # always positive

    # add x and y limits
    #x_lim = [min(min(x), min(x_pole)) - cart_width / 2 - 0.1, max(max(x), max(x_pole)) + cart_width / 2 + 0.1]
    #ylim = [cart_height / 2 - l - 0.05, cart_height / 2 + l + 0.05]

    if (iter>1): # make animation walk forward
        base_x = np.zeros(num_iter_points*iter)
        base_y = np.zeros(num_iter_points*iter)
        base_x[0:num_iter_points] = 0
        base_y[0:num_iter_points] = 0
        for i in range(iter):
            index_prev = i*num_iter_points # end index of cycle i
            index_next = (i+1)*num_iter_points # start index of cycle i
            #if i>0:
            #    base_x[index_prev:index_next] = -x_swing[index_prev] # if cycle is not the first cycle, make new x and y base for stance leg
            #    base_y[index_prev:index_next] = -y_swing[index_prev]

            x_hip[index_prev:index_next] = -l*np.sin(x2[index_prev:index_next]) + base_x[index_prev:index_next]
            y_hip[index_prev:index_next] = l*np.cos(x2[index_prev:index_next]) + base_y[index_prev:index_next]

            x_hip2swing[index_prev:index_next] = l*np.sin(x1[index_prev:index_next])
            y_hip2swing[index_prev:index_next] = -l*np.cos(x1[index_prev:index_next])

            x_swing[index_prev:index_next] = x_hip[index_prev:index_next]+x_hip2swing[index_prev:index_next]
            y_swing[index_prev:index_next] = y_hip[index_prev:index_next]+y_hip2swing[index_prev:index_next]

            base_x[index_next:(index_next+num_iter_points)] = x_swing[index_next-1] 
            base_y[index_next:(index_next+num_iter_points)] = y_swing[index_next-1] 





    fig, ax = plt.subplots()


    def init():

        #init fig and floor
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis("equal")

    def animate(i): 
        ax.clear()
        ax.set_axis_off()
        plt.axis('off')
        #plot slanted floor from initial swing leg pos to final swing leg pos
        ax.plot([0, 0], [0, 0], 'k')
        ax.plot([0, 0.8*iter*0.9], [0, -0.04*iter*0.9], 'k')
        ax.plot([0, 0], [0, 0], 'k')
        ax.plot([0, -0.8], [0, 0.04], 'k')

        if (iter>1): # plot stance leg if cycles>1
            stanceleg = ax.plot([base_x[i], x_hip[i]], [base_y[i], y_hip[i]], 'o-', lw=2, color='green')
        else: # plot stance leg for cycles=1
            stanceleg = ax.plot([0, x_hip[i]], [0, y_hip[i]], 'o-', lw=2, color='green')

        #plot path of the hip
        path_stanceleg = ax.plot(x_hip[:i], y_hip[:i], '--', lw=1, color='green')

        #plot swing leg
        swingleg = ax.plot([x_swing[i], x_hip[i]], [y_swing[i], y_hip[i]], 'o-', lw=2, color='green')

        return swingleg, stanceleg, path_stanceleg

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1), repeat=True, repeat_delay=500)
    if saveFig:
        anim.save(name, writer=animation.PillowWriter(fps=gif_fps))
    #plt.show()


if __name__ == '__main__':

    x1 = np.linspace(-0.3, 0.3, 20)
    x2 = np.linspace(0.3, -0.3, 20)
    phi = 0.0525
    a = 1; b = 1
    animate_compass(x1, x2, a, b, phi, saveFig=True)
