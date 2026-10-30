import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

T_VALUE = 10000

# Applying the Cox-de Boor recursion formula
def bspline_basis(i, p, t, knots): # p- degree
    # Step function calculation
    if p == 0:
        if knots[i] <= t < knots[i + 1]:
            return 1
        # print("value of p in p==0 inside:", p)
        return 0
    # print("value of p in if outside:", p)

    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]

    if denom1 == 0 and denom2 == 0:
        coeff1 = 0
        coeff2 = 0
    elif denom1 == 0:
        coeff1 = 0
        coeff2 = (knots[i + p + 1] - t) /denom2
    elif denom2 == 0:
        coeff1 = (t - knots[i]) / denom1
        coeff2 = 0
    else:
        coeff1 = (t - knots[i]) / denom1
        coeff2 = (knots[i + p + 1] - t) / denom2

    # recursive call of bspline_basis
    # print("value of p",p)
    # Calculated step function going to be used here
    return coeff1 * bspline_basis(i, p - 1, t, knots) + coeff2 * bspline_basis(i + 1, p - 1, t, knots)

def plot_bspline_basis(knots, degree):
    """ Here bspline function is called to plot the Basis function"""
    t = np.linspace(knots[0], knots[-1], T_VALUE, endpoint=False) # It will give T_VALUE = 1000 evenly space points
    # between two Number
    # print("Parameter Value u:",t) # To check the value

    basis = np.zeros((len(knots) - degree - 1, len(t)))
    '''Creating an empty matrix with required row and column'''
    print("shape of final matrix:",basis.shape)  # To check the shape of matrix

    for i in range(len(knots) - degree - 1):  # To decide the row of matrix
        for j in range(len(t)):               # To decide the column of matrix
            basis[i, j] = bspline_basis(i, degree, t[j], knots) # calling bspline_basis function and storing its data
            # print("value of j:",j)                           # j is index of parameter t or u

            # TO CHECK THE SINGLE BASIS FUNCTION MATRIX
        #     if i==0:
        #         print(f"Value of Basis function for i={i} is N{i,degree}:",basis[i, :])
        # if i==0:
        #         # print(basis[i,:])
        #         # print(basis)
        #     break
        # print("outside j")

    print(f"Full Basis Matrix for degree-{degree}:\n",basis) # To check full basis matrix

    plt.figure()
    for i in range(len(knots) - degree - 1):
        plt.plot(t, basis[i, :], label=f'N_{i}_{degree}')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Basis value')
    plt.title(f'B-Spline Basis Functions of degree {degree} with Knot Vector {knots}')
    # plt.grid(True)
    plt.show()

    return basis


# Example usage:
knots = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
# knots = [0,0,0,0,1/4,1/2,3/4,1,1,1,1] #Book Input
'''David F Roger Book Input'''
# knots = [0,1,2,3,4,5,6] # Periodic Knot Vector
# knots = [0,0,0,1,2,2,2]   # Open Uniform knot Vector
# knots = [0,0,0,0.4,2.6,3,3,3]
# knots = [0,0,0,1,1,3,3,3]
# knots = [0,0,0,1.8,2.2,3,3,3]
# knots = [0,0,0,0,5,6,7,8,9,10,11,11,11,11]
# knots = [0,0,0,0,5,6,7,8,9,10,11,12,13,14,15,16,16,16,16]

# ########################################################################

# Calling the Above Function to plot Bspline Basis Function and Storing the Basis Matrix in a Variable
degree = 2
stored_basis_matrix = plot_bspline_basis(knots, degree)

# TO CHECK THE Partition of Unity Property
# def partition_of_unity(basis):
#     t = np.linspace(knots[0], knots[-1], T_VALUE)
#     for j in range(len(t)):
#         sum_of_column = np.sum(basis[:,j])
#         print(f"sum of {j} column of Basis Matrix", sum_of_column)
#
# partition_of_unity(stored_basis_matrix)