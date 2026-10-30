from tkinter import *
import matplotlib.pyplot as plt
from tkinter.ttk import *

def clear_input():
    for widget in root.winfo_children():
        if isinstance(widget, Entry):
            widget.delete(0, 'end')

def input_order():
    order = int(order_input.get())
    print(order)
    list_x = []
    list_y = []

    def cp_input1():
        for n in list_x:
            list_y.append(int(n.get()))
            print(list_y)

    def first_degree():
        u = 0
        t_x = []
        t_y = []
        t_z = []
        while u <= 1:
            put_x = pow(1 - u, 3) * list_y[0] + 3 * u * pow(1 - u, 2) * list_y[3] + 3 * u * u * pow(1 - u, 1) * \
                    list_y[6] \
                    + pow(u, 3) * list_y[9]
            put_y = pow(1 - u, 3) * list_y[1] + 3 * u * pow(1 - u, 2) * list_y[4] + 3 * u * u * pow(1 - u, 1) * list_y[7] \
                    + pow(u, 3) * list_y[10]
            put_z = pow(1 - u, 3) * list_y[2] + 3 * u * pow(1 - u, 2) * list_y[5] + 3 * u * u * pow(1 - u, 1) * list_y[
                8] \
                    + pow(u, 3) * list_y[11]

            t_x.append(put_x)
            t_y.append(put_y)
            t_z.append(put_z)
            u += 0.001
        # To separate x and y co-ordinates
        cod_x = []
        cod_y = []
        cod_z = []
        g=0
        h=1
        k=2
        for t in range(len(list_y)-8):
            cod_x.append(list_y[g])
            cod_y.append(list_y[h])
            cod_z.append(list_y[k])
            g+=3
            h+=3
            k+=3
        print(cod_x, cod_y, cod_z)

        ax = plt.axes(projection='3d')
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")

        ax.plot3D(t_x, t_y, t_z, 'black')
        ax.scatter(cod_x, cod_y, cod_z, c='Red')
        ax.set_title('3D Bezier plot for n=3')
        plt.show()

    if order == 3:
        for i in range(1, order + 2):
            cp_label = Label(text=f"Enter {i}st CP:", font=('Arial', 10, "bold"))
            cp_label.grid(column=0, row=i,padx=10, pady=10)

        for x in range(order + 1):
            for j in range(order):
                cp_input = Entry()
                cp_input.grid(column=j + 1, row=x + 1)
                list_x.append(cp_input)

        ok_button1 = Button(text="ok", command=cp_input1)
        ok_button1.grid(column=4, row=2)

        clear = Button(root, text="Clear", command=clear_input)
        clear.grid(column=1, row=5)

        draw_button = Button(text="Draw", command=first_degree)
        draw_button.grid(column=2, row=5)


# GUI Part

root = Tk()
root.minsize(width=600, height=600)
root.title("Bezier Curve")
root.config(pady=20, padx=20)

# For order interface
order_label = Label(text="Enter Degree", font=('Arial', 10, "bold"))
order_label.grid(column=0, row=0)
order_input = Entry()
order_input.grid(column=1, row=0)
ok_button = Button(text="ok", command=input_order)
ok_button.grid(column=2, row=0)

root.mainloop()
