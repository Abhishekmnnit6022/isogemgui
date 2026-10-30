import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')

# USER DEFINED VARIABLES:

# Control Points
x = np.array([[-0.5, -2, 0],
              [1, 1, 1],
              [2, 2, 2]])

y = np.array([[2, 1, 0],
              [2, 8, -1],
              [2, 1, 1]])

z = np.array([[1, -1, 2],
              [0, -8.5, 2],
              [0.5, 1, 2]])

# For Rectangle Surface
# x = np.array([[0,2,3],[0,2,3],[0,2,3]])
# y = np.array([[0,0,0],[1,1,1],[2,2,2]])
# z = np.array([[0,0,0],[0,0,0],[0,0,0]])

# Book Input
# x = np.array([[-15, -5, 5,15],[-15, -5, 5,15],[-15, -5, 5,15],[-15, -5, 5,15]])
# y = np.array([[0,5,5,0],[5,5,5,5], [5,5,5,5],[0,5,5,0]])
# z = np.array([[15,15,15,15],[5,5,5,5], [-5,-5,-5,-5],[-15,-15,-15,-15]])


# DEPENDANT VARIABLES
# Total Number of Control Points in u and v direction
uPTS = np.size(x, 0)
vPTS = np.size(x, 1)

# Total Number of Subdivisions or Deciding Degree in each direction
n = uPTS - 1  #
m = vPTS - 1  #

# Parametric Variable
# Number of Cells for Each Direction
uCELLS = 100
vCELLS = 100
u = np.linspace(0, 1, uCELLS)
print("Value of u:", u)
v = np.linspace(0, 1, vCELLS)

# To Bernstein Basis Polynomial, this b, d value is used to plot basis function
b = []
d = []

# Initialized Empty Matrix for X, Y, and Z, Bézier curve
xBezier = np.zeros((uCELLS, vCELLS))
yBezier = np.zeros((uCELLS, vCELLS))
zBezier = np.zeros((uCELLS, vCELLS))


# Binomial Coefficients
def Ni(n, i): # n=3, i=0,1,2
    return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))


def Mj(m, j):
    return np.math.factorial(m) / (np.math.factorial(j) * np.math.factorial(m - j))


# Bernstein Basis Polynomial
def J(n, i, u):  # n=3, i=0 , u= [0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
    # 0.66666667 0.77777778 0.88888889 1.    ]
    return np.matrix(Ni(n, i) * (u ** i) * (1 - u) ** (n - i))


def K(m, j, w):
    return np.matrix(Mj(m, j) * (w ** j) * (1 - w) ** (m - j))


# MAIN LOOP

for i in range(0, uPTS):  # example 3 degree tak chalega, i = 0,1,2
    for j in range(0, vPTS):  # 3 bar chalega, j = 0,1,2
        # Stores individual Basis functions, b,d use to plot basis function
        b.append(J(n, i, u))
        d.append(K(m, j, v))

        # Transpose J array
        # print("Size of J:",J)
        # first loop n=3, i=0,1,2 , u= []

        J_transpose = J(n, i, u).transpose()  # First loop run, n=3 fixed , i=0 fixed, and u ke sare value ke liye
        # calculate hoga

        print("size of J_Trans:", J_transpose.shape)

        # print("Value of J_Trans:", J_transpose)

        # Bezier Curve Calculation

        xBezier = J_transpose * K(m, j, v) * x[i, j] + xBezier
        yBezier = J_transpose * K(m, j, v) * y[i, j] + yBezier
        zBezier = J_transpose * K(m, j, v) * z[i, j] + zBezier

        k_matrix = K(m, j, v)
        print("Value of k_matrix:", k_matrix)
        print("size of k_:", k_matrix.shape)

        shape_of_xBezier = xBezier
        # print("Value of x_Bezier:", xBezier)
        print("Size of x_Bezier:", xBezier.shape)

# plotting
# plt.figure()
# plt.subplot(121)
# for line in b:
#     plt.plot(u, line.transpose())
# plt.subplot(122)
# for line in d:
#     plt.plot(v, line.transpose())
# plt.show()

# Bezier Surface
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Bezier Rectangular Surface Plot')

plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
ax.plot_surface(xBezier, yBezier, zBezier, cmap='viridis')
ax.scatter(x, y, z, edgecolors='face')
plt.show()
