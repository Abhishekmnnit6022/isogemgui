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

    def second_degree():
        u = 0
        t_x = []
        t_y = []
        while u <= 1:
            put_x = pow(1 - u, 2) * list_y[0] + 2 * u * pow(1 - u, 1) * list_y[2] + pow(u, 2) * list_y[4]
            put_y = pow(1 - u, 2) * list_y[1] + 2 * u * pow(1 - u, 1) * list_y[3] + pow(u, 2) * list_y[5]
            t_x.append(put_x)
            t_y.append(put_y)
            u += 0.001
        cod_x = []
        cod_y = []
        g=0
        h=1
        for t in range(len(list_y)-3):
            cod_x.append(list_y[g])
            cod_y.append(list_y[h])
            g+=2
            h+=2
        print(cod_x, cod_y)
        name = ["First Order Bezier Curve", "Second Degree Bezier Curve", "n=3"]
        plt.title(name[order - 1])
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.plot(cod_x, cod_y, 'o')
        plt.plot(cod_x, cod_y, color='b', label='control Polygon')
        plt.plot(t_x, t_y, color='g', label='curve')
        plt.show()

    if order ==2:
        cp_label = Label(root, text="Enter 1st cp", font=('Arial', 12, "bold"))
        cp_label.grid(column=0, row=1,padx=10, pady=10)

        cp_label = Label(root, text="Enter 2nd cp", font=('Arial', 12, "bold"))
        cp_label.grid(column=0, row=2,padx=10, pady=10)

        cp_label = Label(root, text="Enter 3rd cp", font=('Arial', 12, "bold"))
        cp_label.grid(column=0, row=3, padx=10, pady=10)

        for x in range(order+1):
            for j in range(order):
                cp_input = Entry(root,font=('Arial', 11))
                cp_input.grid(column=j+1, row=x+1)
                list_x.append(cp_input)

        ok_button1 = Button(root, text="ok", command=cp_input1)
        ok_button1.grid(column=3, row=2)

        clear = Button(root, text="Clear", command=clear_input)
        clear.grid(column=1, row=4)

        draw_button = Button(root, text="Draw", command=second_degree)
        draw_button.grid(column=2, row=4)


# GUI Part

root = Tk()
root.minsize(width=600, height=500)
root.title("Bezier Curve")
root.config(pady=20, padx=20)

# For order interface
order_label = Label(text="Enter Degree", font=('Arial', 12, "bold"))
order_label.grid(column=0, row=0, padx=10, pady=10)
order_input = Entry(root,font=('Arial', 11))
order_input.grid(column=1, row=0)
ok_button = Button(text="ok", command=input_order)
ok_button.grid(column=2, row=0)


root.mainloop()
