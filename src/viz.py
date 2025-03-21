from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import matplotlib.pyplot as plt
import numpy as np

L = 10

# Read file and split timestemps by two newlines
with open("particles") as f:
    data = f.read().strip().split("\n\n")

# process timestemps
timesteps = []
for t, timestep in enumerate(data):
    lines = timestep.splitlines()
    particle_positions = []

    for line in lines:
        line = [d.strip() for d in line.split(',')]
        r = np.array([float(d) for d in line[1:]])

        if np.any(r > L):
            print(f"dropping particle @ {t}")
        else:
            timesteps[t].append(r)

    if particle_positions:
        timesteps.append(np.stack(particle_positions))

timesteps = timesteps[:200]

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
ani.save('md.gif', writer='imagemagick')
plt.show()