#!python3

import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import matplotlib.pyplot as plt
import numpy as np

def main(argc, argv):
    if argc != 4:
        print(f"Usage: {argv[0]} axis_length input.csv output.gif")
        return

    L = int(argv[1])

    # Read file and split timestemps by two newlines
    with open(argv[2]) as f:
        f.readline()
        data = f.read().strip().split("\n\n")

    # process timestemps
    timesteps = []
    for t, timestep in enumerate(data):
        particles = timestep.splitlines()
        particle_positions = []

        for particle in particles:
            particle = [d.strip() for d in particle.split(',')]
            # coord in 3D space
            r = np.array([float(d) for d in particle[1:]])

            particle_positions.append(r)

        if particle_positions:
            timesteps.append(np.stack(particle_positions))

    #timesteps = timesteps[:200]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def update(t, ax):
        print(f"Animating timestep {t}")
        ax.clear()
        xs, ys, zs = np.split(timesteps[t],3,1)
        ax.scatter(xs, ys, zs, c="r")
        ax.set_xlim3d([0.0, L])
        ax.set_xlabel('X')

        ax.set_ylim3d([0.0, L])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, L])
        ax.set_zlabel('Z')



    # Setting the axes properties
    ani = animation.FuncAnimation(fig, update, len(timesteps), fargs=(ax,), interval=10, repeat_delay=5000, blit=False)
    ani.save(argv[3], writer='imagemagick')
    plt.show()

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)