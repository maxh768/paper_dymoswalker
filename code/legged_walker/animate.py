import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

def animate(L, a1, a2, b1, b2, q1, q2):
    # L, a1, a2, b1, b2 are fixed lengths, q1 and q2 are angles of legs (vectors)
    fig, ax = plt.subplots()

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis("equal")