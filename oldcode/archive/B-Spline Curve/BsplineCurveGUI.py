from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

T_VALUE = 100

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

    return coeff1 * bspline_basis(i, p - 1, t, knots) + coeff2 * bspline_basis(i + 1, p - 1, t, knots)


def plot_bspline_basis():
    knots_str = knots_entry.get()
    knots = [float(knot.strip()) for knot in knots_str.split(",")]
    degree = int(degree_entry.get())

    t = np.linspace(knots[0], knots[-1], T_VALUE, endpoint=True)
    basis = np.zeros((len(knots) - degree - 1, len(t)))

    for i in range(len(knots) - degree - 1):
        for j in range(len(t)):
            basis[i, j] = bspline_basis(i, degree, t[j], knots)

    fig = plt.figure()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=3, columnspan=len(knots))

    for i in range(len(knots) - degree - 1):
        plt.plot(t, basis[i, :], label=f'N_{i}_{degree}')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Basis value')
    plt.title(f'B-Spline Basis Functions of degree {degree} with Knot Vector {knots}')

    canvas.draw()

root = tk.Tk()
root.title("B-Spline Basis Function Plotter")

# Create style for text label
style = ttk.Style(root)
style.configure("TLabel", font=("Arial", 12))

# Create knot input field
knot_label = ttk.Label(root, text="Knots (comma-separated):", style = "TLabel")
knot_label.grid(row=0, column=0, sticky='E')
knots_entry = ttk.Entry(root, font=("Arial", 12))
knots_entry.grid(row=0, column=1, padx=5, pady=5)

# Create degree input field
degree_label = ttk.Label(root, text="Degree:", style="TLabel")
degree_label.grid(row=1, column=0, sticky='E')
degree_entry = ttk.Entry(root, font=("Arial", 12))
degree_entry.grid(row=1, column=1, padx=5, pady=5)

# Create style for button
style.configure("TButton", font=("Arial", 14, "bold"))
# Create plot button
plot_button = ttk.Button(root, text="Plot", style="TButton", command=plot_bspline_basis)
plot_button.grid(row=2, columnspan=2, pady=10)

root.mainloop()
