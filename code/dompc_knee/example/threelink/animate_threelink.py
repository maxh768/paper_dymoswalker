import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

def animate_threelink(x1, x2, x3, a1, b1, a2, b2, phi, name='compass.gif', interval = 20, saveFig=False, gif_fps=20, iter=1, len_iterarr = []):
    ls = a1 + b1
    lt = a2 + b2
    l = ls + lt

    """MATH"""
    # path of stance leg tip (hip)
    x_hip = -l*np.sin(x1)
    y_hip = l*np.cos(x1)

    # intermediate coords of knee relative to hip:
    x_hip2knee = lt*np.sin(x2)
    y_hip2knee = -lt*np.cos(x2) # may need to look at signs

    # real coords of knee:
    x_knee = x_hip+x_hip2knee
    y_knee = y_hip + y_hip2knee

    # intermediate coords of swing relative to knee:
    x_knee2swing = ls*np.sin(x3)
    y_knee2swing = -ls*np.cos(x3)

    # real coords of swing leg path
    x_swing = x_knee + x_knee2swing
    y_swing = y_knee + y_knee2swing


    print(iter)
    print(len_iterarr)
    if(iter > 1):
        total_points = int(np.sum(len_iterarr))
        print('total points :', total_points)
        base_x = np.zeros(total_points)
        base_y = np.zeros(total_points)
        for i in range(iter):
            cur_len = int(len_iterarr[i])
            index_prev = int(sum(len_iterarr[:i]))
            index_next = index_prev + cur_len
            print('index prev:', index_prev)
            print('index next:', index_next)

            x_hip[index_prev:index_next] = -l*np.sin(x1[index_prev:index_next]) + base_x[index_prev:index_next] 
            y_hip[index_prev:index_next] = l*np.cos(x1[index_prev:index_next]) + base_y[index_prev:index_next]

            x_hip2knee[index_prev:index_next] = lt*np.sin(x2[index_prev:index_next])
            y_hip2knee[index_prev:index_next] = -lt*np.cos(x2[index_prev:index_next])

            x_knee[index_prev:index_next] = x_hip[index_prev:index_next]+x_hip2knee[index_prev:index_next]
            y_knee[index_prev:index_next] = y_hip[index_prev:index_next]+y_hip2knee[index_prev:index_next]

            x_knee2swing[index_prev:index_next] = ls*np.sin(x3[index_prev:index_next])
            y_knee2swing[index_prev:index_next] = -ls*np.cos(x3[index_prev:index_next])

            x_swing[index_prev:index_next] = x_knee[index_prev:index_next] + x_knee2swing[index_prev:index_next]
            y_swing[index_prev:index_next] = y_knee[index_prev:index_next] + y_knee2swing[index_prev:index_next]


            if i == iter-1:
                next_len = int(len_iterarr[iter-1])
            else:
                next_len = int(len_iterarr[i+1])
            base_x[index_next:(index_next+next_len)] = x_swing[index_next-1]
            base_y[index_next:(index_next+next_len)] = y_swing[index_next-1] 


            


    """ANIMATE"""
    print('xhip: ', len(x_hip))
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
        #plot slanted floor from initial swing leg pos to final swing leg pos
        ax.plot([0, 0], [0, 0], 'k')
        ax.plot([0, 0.8*iter*0.9], [0, -0.04*iter*0.9], 'k')
        ax.plot([0, 0], [0, 0], 'k')
        ax.plot([0, -0.8], [0, 0.04], 'k')


        if(iter>1):
            stanceleg = ax.plot([base_x[i], x_hip[i]], [base_y[i], y_hip[i]], 'o-', lw=2, color='green')
        else:
            stanceleg = ax.plot([0, x_hip[i]], [0, y_hip[i]], 'o-', lw=2, color='green')
        #plot path of the hip
        path_stanceleg = ax.plot(x_hip[:i], y_hip[:i], '--', lw=1, color='green')

        #plot thigh
        thigh = ax.plot([x_knee[i], x_hip[i]], [y_knee[i], y_hip[i]], 'o-', lw=2, color='green')

        #plot swing leg
        swing = ax.plot([x_swing[i], x_knee[i]], [y_swing[i], y_knee[i]], 'o-', lw=2, color='green')

        return swing, stanceleg, path_stanceleg, thigh

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1), repeat=True, repeat_delay=500)
    if saveFig:
        anim.save(name, writer=animation.PillowWriter(fps=gif_fps))
    #plt.show()


if __name__ == '__main__':

    x1 = np.linspace(-0.3, 0.3, 20)
    x2 = np.linspace(0.3, -0.3, 20)
    phi = 0.0525
    a = 1; b = 1
    #animate_threelink(x1, x2, a, b, phi, saveFig=True)
