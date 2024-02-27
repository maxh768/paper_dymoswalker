import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation


def animate_compass(x1, x2, a, b, phi, interval = 20, saveFig=False, gif_fps=20):
    #x1: back leg
    #x2: front leg
    #phi: angle of slope

    l = a + b

    extend = 500 // interval
    x1 = np.concatenate((x1, np.ones(extend) * x1[-1]))
    x2 = np.concatenate((x2, np.ones(extend) * x2[-1]))

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

    fig, ax = plt.subplots()

    def init():

        #init fig and floor
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis("equal")

    def animate(i): 
        ax.clear()

        #plot slanted floor from initial swing leg pos to final swing leg pos
        ax.plot([0, x_swing[1]], [0, y_swing[1]], 'k')
        xswinglen = len(x_swing) - 1
        ax.plot([0, x_swing[xswinglen]], [0, y_swing[xswinglen]], 'k')

        # plot stance leg
        stanceleg = ax.plot([0, x_hip[i]], [0, y_hip[i]], 'o-', lw=2, color='C0')

        #plot path of the hip
        path_stanceleg = ax.plot(x_hip[:i], y_hip[:i], '--', lw=1, color='black')

        #plot swing leg
        swingleg = ax.plot([x_swing[i], x_hip[i]], [y_swing[i], y_hip[i]], 'o-', lw=2, color='purple')

        return swingleg, stanceleg, path_stanceleg

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1), interval=interval, repeat=True)
    if saveFig:
        anim.save("compass.gif", writer=animation.PillowWriter(fps=gif_fps))
    #plt.show()


if __name__ == '__main__':

    x1 = np.linspace(-0.3, 0.3, 20)
    x2 = np.linspace(0.3, -0.3, 20)
    phi = 0.0525
    a = 1; b = 1
    animate_compass(x1, x2, a, b, phi, saveFig=True)
