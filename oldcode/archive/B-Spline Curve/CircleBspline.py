import numpy as np
import matplotlib.pyplot as plt

T_VALUE = 900

def bspline_basis(i, p, t, knots):
    if p == 0:
        if knots[i] <= t < knots[i + 1]:
            return 1
        return 0

    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]

    if denom1 == 0 and denom2 == 0:
        coeff1 = 0
        coeff2 = 0
    elif denom1 == 0:
        coeff1 = 0
        coeff2 = (knots[i + p + 1] - t) / denom2
    elif denom2 == 0:
        coeff1 = (t - knots[i]) / denom1
        coeff2 = 0
    else:
        coeff1 = (t - knots[i]) / denom1
        coeff2 = (knots[i + p + 1] - t) / denom2

    # recursive call of bspline_basis
    return coeff1 * bspline_basis(i, p - 1, t, knots) + coeff2 * bspline_basis(i + 1, p - 1, t, knots)


def plot_bspline_basis(knots, degree):
    t = np.linspace(knots[0], knots[-1], T_VALUE)
    basis = np.zeros((len(knots) - degree - 1, len(t)))

    for i in range(len(knots) - degree - 1):
        for j in range(len(t)):
            basis[i, j] = bspline_basis(i, degree, t[j], knots)

    plt.figure()
    for i in range(len(knots) - degree - 1):
        plt.plot(t, basis[i, :], label=f'N_{i}_{degree}')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Basis value')
    plt.title(f'B-Spline Basis Functions of degree {degree} with Non-Uniform Knot Vector')
    plt.show()
    # print(basis)

    return basis


def plot_bspline_curve(basis, control_points):
    t = np.linspace(0, 1, T_VALUE)
    curve_points = np.zeros((control_points.shape[0], len(t)))  #shape[row i.e 0] 2 row and 100(len of t) column [2,100]
    # curve_points = np.zeros((2, len(t)))
    for i in range(len(t)):
        for j in range(control_points.shape[1]):  # shape[1] gives total no. fo column in control points array
            curve_points[:, i] += basis[j, i] * control_points[:, j]

    plt.figure()
    plt.plot(control_points[0, :], control_points[1, :], 'bo-', label='Control Points')
    plt.plot(curve_points[0, :], curve_points[1, :], 'r-', label='B-Spline Curve')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'B-Spline Curve having Degree {degree} and Order {degree+1} with Control Points')
    plt.show()
    # print(curve_points)


# Example usage:
# knots = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 4.9995]
knots = [0, 0, 0, 1/4, 1/4, 1/2, 1/2, 3/4, 3/4, 1, 1, 1]

degree = 3

# Generate basis matrix
basis_matrix = plot_bspline_basis(knots, degree)

#  Circular Control points
number_of_control_points = len(knots)-degree-1
print(number_of_control_points)
angle = np.linspace(0, 2*np.pi,number_of_control_points)  # Divide the circle into 7 equal angles
radius = 0.5  # Radius of the circle

x = radius * np.cos(angle)
y = radius * np.sin(angle)

control_points = np.array([x, y])
print("cp array:", control_points)
# control_points = np.array(control_points_list, dtype=np.float64)

# Plot B-spline curve
plot_bspline_curve(basis_matrix, control_points)
