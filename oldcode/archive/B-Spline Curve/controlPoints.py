import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

# Control points for a circle
angle = np.linspace(0, 2*np.pi, 10)  # Divide the circle into 7 equal angles
radius = 1.0  # Radius of the circle

x = radius * np.cos(angle)
y = radius * np.sin(angle)

control_points = np.array([x, y])
# print(control_points)
control_points_list = control_points.tolist()
print("Control points in list:\n", control_points_list)
print(type(control_points_list[0][0]))

# Plotting the control points
plt.plot(control_points[0, :], control_points[1, :], 'bo-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Control Points Forming a Circle')
plt.axis('equal')
plt.grid(True)
plt.show()

