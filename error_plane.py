import numpy as np
import matplotlib.pyplot as plt

show_axes = False

# generate data for a hilly landscape
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = 0.5 * np.sin(0.5 * x) * np.cos(0.5 * y) + 0.3 * np.sin(x) * np.cos(y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, cmap='gist_earth')

fig.colorbar(surf)

if show_axes:
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
else:
    ax.set_axis_off()

z_max = np.max(z)
ax.set_zlim(-z_max, 5 * z_max)

plt.show()
