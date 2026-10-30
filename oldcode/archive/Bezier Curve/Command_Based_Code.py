import matplotlib.pyplot as plt
from numpy import *
import matplotlib
matplotlib.use('tkagg')

# CREATING EMPTY ARRAY FOR STORING CONTROL POINT INPUT
array_x = array([])
array_y = array([])

order = int(input(f"Enter No. of Degree :"))

for i in range(order + 1):
    value_x = int(input(f"Enter x  coordinate of {i}th control point: "))
    array_x = append(array_x, value_x)
    value_y = int(input(f"Enter y coordinate of {i}th control point: "))
    array_y = append(array_y, value_y)

x = array([])
y = array([])

u = 0
while u <= 1:
    if order == 1:
        put_x = (1 - u) * array_x[0] + u * array_x[1]
        put_y = (1 - u) * array_y[0] + u * array_y[1]
        x = append(x, put_x)
        y = append(y, put_y)

    if order == 2:
        put_x = pow(1 - u, 2) * array_x[0] + 2 * u * pow(1 - u, 1) * array_x[1] + pow(u, 2) * array_x[2]
        put_y = pow(1 - u, 2) * array_y[0] + 2 * u * pow(1 - u, 1) * array_y[1] + pow(u, 2) * array_y[2]
        x = append(x, put_x)
        y = append(y, put_y)

    if order == 3:
        put_x = pow(1 - u, 3) * array_x[0] + 3 * u * pow(1 - u, 2) * array_x[1] + 3 * u * u * pow(1 - u, 1) * array_x[2] \
                + pow(u, 3) * array_x[3]
        put_y = pow(1 - u, 3) * array_y[0] + 3 * u * pow(1 - u, 2) * array_y[1] + 3 * u * u * pow(1 - u, 1) * array_y[2] \
                + pow(u, 3) * array_y[3]
        x = append(x, put_x)
        y = append(y, put_y)

    u += 0.001
    # print(x)
name = ["First Degree Bezier Curve", "Second Degree Bezier Curve", "Cubic Bezier Curve"]
plt.title(name[order - 1])
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.plot(array_x, array_y, 'o')
plt.plot(array_x, array_y, color='b', label='control Polygon')
plt.plot(x, y, color='g', label='curve')
plt.legend()
plt.show()

# How TO DRAW POINTS ON PYTHON

# import matplotlib.pyplot as plt
# x = range(1,10)
# y = range(1,10)
# plt.plot(x,y,'o')
# plt.show()
