import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')

T_VALUE = 10000


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
    t = np.linspace(knots[0], knots[-1], T_VALUE, endpoint=False)  # This t is ui
    basis = np.zeros((len(knots) - degree - 1, len(t)))

    for i in range(len(knots) - degree - 1):
        for j in range(len(t)):
            basis[i, j] = bspline_basis(i, degree, t[j], knots)

    plt.figure()
    for i in range(len(knots) - degree - 1):
        plt.plot(t, basis[i, :], label=f'N_{i}_{degree}')

    plt.legend()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('Basis value', fontsize=20)
    plt.title(f'B-Spline Basis Functions of degree {degree} with knot Vector {knots}')
    plt.show()
    print("Basis Matrix:", basis)

    return basis


def plot_bspline_curve(basis, control_points):
    t = np.linspace(0, 1, T_VALUE, endpoint=False)  # This t is u
    curve_points = np.zeros((control_points.shape[0], len(t))) # shape[row i.e 0] 2 row and 100(len of t) column [2,100]
    # curve_points = np.zeros((2, len(t)))
    for i in range(len(t)):
        for j in range(control_points.shape[1]):  # shape[1] gives total no. fo column in control points array
            curve_points[:, i] += basis[j, i] * control_points[:, j]

    plt.figure()
    plt.plot(control_points[0, :], control_points[1, :], 'bo-', label='Control Points')  # plot control point polygon
    plt.plot(curve_points[0, :], curve_points[1, :], 'r-', label='B-Spline Curve')  # plot bspline curve
    plt.legend()  # plot both curve on same window
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.title(f'B-Spline Curve having Degree {degree} for knots {knots}')  # just for title
    plt.show()  # to show the window on screen
    print("Bspline Curve Point Value of x,y:\n", curve_points)


# Example usage:
# knots = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
# knots = [1,2,3,4,5,6,7,8]
######################## David F. Roger Book ###########################################
# knots = [0,1,2,3,4,5,6,7] # Periodic not working
# control_points = np.array([[31,5,9,56,23],
#                            [11,6,25,20,36]])
# degree = 2
# knots = [0,0,0,1,2,3,3,3]  # Working for Open Uniform
# control_points = np.array([[31,5,9,56,23],[11,6,25,20,36]])

# knots = [0,0,0,1,1,2,2,2]    # Working for
# control_points = np.array([[31,5,9,56,23],[11,6,25,20,36]])

# knots = [0,0,0,1,2,3,3,3]
# control_points = np.array([[31,5,9,56],[11,6,25,20]])


# knots = [0,0,0,0.4,2.6,3,3,3] # Working for control_points = np.array([[31,5,9,56,23],[11,6,25,20,36]])

# knots = [0,0,0,1.8,2.2,3,3,3]

######################## THE NURBS BOOK PAGE 88 ###########################################

# knots = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Use degree = 9, knots for Bezier Curve
# knots = [0,0,0,1/8,2/8,3/8,4/8,5/8,6/8,7/8,1,1,1]  # Use degree = 2, CP remain same
# control_points = np.array([[5, 8, 10, 15, 20, 30, 35, 40, 45, 50],
#                            [5, 8, 5, 15, 13, 1, 10, 7, 12, 11]])

# knots=[0,0,0,0,1/4,1/2,3/4,3/4,1,1,1,1]              # use degree = 3
# control_points = np.array([[2,3,10,20,18,25,30,24],
#                            [3,7,10,8,3,3,4,9]])
# degree = 9

########################################## My Example ##################################################
# knots = [0,0,0,0,2.4,2.8,3.2,3.6,4.0,4.4,4.8,5.2,5.6,6.0,6.4,16,16,16,16]
# degree = 3
# control_points = np.array([[5, 7, 10, 25, 20, 30, 35, 40, 45, 50, 70, 20, 80, 100, 95],
#                            [5, 8, 5, 25, 13, 1, 10, 7, 12, 11, 20, 35, 40, 90, 53]])
#########################################################################################################
# Generate basis matrix
basis_matrix = plot_bspline_basis(knots, degree)

# Control points
# control_points = np.array([[1, 0,-1,0],
#                            [0, 1,0,-1]])

#  Circular Control points
# Example -01
# control_points = np.array([[1.00000000e+00, 6.23489802e-01, -2.22520934e-01, -9.00968868e-01,
#                             -9.00968868e-01, -2.22520934e-01, 6.23489802e-01, 1.00000000e+00],
#                            [0.00000000e+00, 7.81831482e-01, 9.74927912e-01, 4.33883739e-01,
#                             -4.33883739e-01, -9.74927912e-01, -7.81831482e-01, -2.44929360e-16]])

# Example -02, from T. Hughes Paper
# control_points = np.array([[5, 5, 50, 50, 70, 80, 90, 100],
#                            [20, 70, 2, 40, 40, 2, 30, 30]])

# Plot B-spline curve
plot_bspline_curve(basis_matrix, control_points)
