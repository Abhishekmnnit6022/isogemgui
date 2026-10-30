from scipy.spatial import ConvexHull
from Command_Based import *

# Control points
x = np.array([[-0.5, -2, 0], [1, 1, 1], [2, 2, 2]])
y = np.array([[2, 1, 0], [2, 8, -1], [2, 1, 1]])
z = np.array([[1, -1, 2], [0, -8.5, 2], [0.5, 1, 2]])

# Book Input
# x = np.array([[-15, -5, 5,15],[-15, -5, 5,15],[-15, -5, 5,15],[-15, -5, 5,15]])
# y = np.array([[0,5,5,0],[5,5,5,5], [5,5,5,5],[0,5,5,0]])
# z = np.array([[15,15,15,15],[5,5,5,5], [-5,-5,-5,-5],[-15,-15,-15,-15]])

# Compute convex hull
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
hull = ConvexHull(points)

# Create a 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot surface

ax.plot_surface(xBezier, yBezier, zBezier, alpha=0.5)

# Plot control points
ax.scatter(x, y, z, edgecolors='face', c='red', marker='o')

# Plot convex hull
for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'b-')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Control Points and Convex Hull')

# Show the plot
plt.show()
