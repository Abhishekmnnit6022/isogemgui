import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

def bernstein_poly(i, n, u):
    """
    The Bernstein polynomial of degree n, for index i, evaluated at u.
    """
    return np.math.comb(n, i) * u**i * (1 - u)**(n - i)

def bezier_curve(control_points, n_points=1000):
    """
    Given a set of control points, return the corresponding Bézier curve.
    """
    n = len(control_points) - 1    # Degree= Control points -1
    curve_points = []
    for u in np.linspace(0, 1, n_points):
        # Calculate the value of the Bézier curve at u.
        point = np.zeros((2,))
        for i, p in enumerate(control_points):
            point += bernstein_poly(i, n, u) * p
        curve_points.append(point)
    return np.array(curve_points)

# Define a function to get the control points from the user using a Tkinter GUI.
def get_control_points():
    control_points = []
    def add_point():
        x, y = float(x_entry.get()), float(y_entry.get())
        control_points.append((x, y))
        print(control_points)
        points_label.config(text=f"Control points: {control_points}")
        x_entry.delete(0, tk.END)
        y_entry.delete(0, tk.END)
    def clear_points():
        nonlocal control_points
        control_points = []
        points_label.config(text=f"Control points: {control_points}")
    def generate_curve():
        curve_points = bezier_curve(np.array(control_points))
        plt.plot(np.array(control_points)[:, 0], np.array(control_points)[:, 1], 'bo-')
        plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-')
        plt.xlabel("X-Axis", fontsize=13)
        plt.ylabel("Y-Axis", fontsize=13)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.show()

    root = tk.Tk()
    root.title("Enter control points")
    tk.Label(root, text="Enter (x, y) coordinates of control points:").grid(row=0, column=0, columnspan=2)
    tk.Label(root, text="x:").grid(row=1, column=0)
    tk.Label(root, text="y:").grid(row=2, column=0)
    x_entry = tk.Entry(root)
    y_entry = tk.Entry(root)
    x_entry.grid(row=1, column=1)
    y_entry.grid(row=2, column=1)
    tk.Button(root, text="Add point", command=add_point).grid(row=3, column=0)
    tk.Button(root, text="Clear points", command=clear_points).grid(row=3, column=1)
    points_label = tk.Label(root, text="Control points: []")
    points_label.grid(row=4, column=0, columnspan=2)
    tk.Button(root, text="Generate curve", command=generate_curve).grid(row=5, column=0, columnspan=2)
    tk.mainloop()

# Call the get_control_points function to start the GUI.
get_control_points()
