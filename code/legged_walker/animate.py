import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

def animate(L, a1, a2, b1, b2, q1, q2):
    # L, a1, a2, b1, b2 are fixed lengths, q1 and q2 are angles of legs (vectors)

    ln = len(q1)
    fig, ax = plt.subplots()
    
    #x_backleg = np.ones(ln)
    #y_backleg = np.ones(ln)

    y_fixed = L*np.cos(q2)
    x_fixed = L*np.sin(q2)

    #x_frontleg = np.ones(ln)
    #y_frontleg = np.ones(ln)
    



    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis("equal")