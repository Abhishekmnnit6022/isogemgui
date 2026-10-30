from tkinter import *
import matplotlib.pyplot as plt
from tkinter.ttk import *
import matplotlib
matplotlib.use('tkagg')

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
            list_y.append(float(n.get()))
            print(list_y)

    def first_degree():
        u = 0
        t_x = []
        t_y = []
        while u <= 1:
            put_x = (1 - u) * list_y[0] + u * list_y[1]
            put_y = (1 - u) * list_y[2] + u * list_y[3]
            t_x.append(put_x)
            t_y.append(put_y)
            u += 0.001
        name = ["First Degree Bezier", "Second Order Bezier", "Cubic Bezier Curve"]
        plt.title(name[order - 1])
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")

        plt.plot()
        plt.plot(t_x, t_y, color='g', label='curve')
        plt.show()

    if order ==1:
        cp_label = Label(text="Enter 1st cp:", font=('Arial', 11, "bold"))
        cp_label.grid(column=0, row=1)

        cp_label = Label(text="Enter 2st cp:", font=('Arial', 11, "bold"))
        cp_label.grid(column=0, row=2)

        for x in range(order+1):
            cp_input = Entry(root,font=('Arial', 11))
            cp_input.grid(column=1+x, row=1)
            list_x.append(cp_input)

            cp_input = Entry(root,font=('Arial', 11))
            cp_input.grid(column=1+x, row=2)
            list_x.append(cp_input)

        ok_button1 = Button(text="ok", command=cp_input1)
        ok_button1.grid(column=3, row=2)

        clear = Button(root, text="Clear", command=clear_input)
        clear.grid(column=1, row=4)

        draw_button = Button(text="Draw", command=first_degree)
        draw_button.grid(column=1, row=3)


# GUI Part

root = Tk()
root.minsize(width=500, height=500)
root.title("Bezier Curve")
root.config(pady=20, padx=20)

# For order interface

order_label = Label(text="Enter Degree:", font=('Arial', 11, "bold"),foreground='black')
order_label.grid(column=0, row=0)
order_input = Entry(root,font=('Arial', 11))
order_input.grid(column=1, row=0)
ok_button = Button(text="ok", command=input_order)
ok_button.grid(column=2, row=0)


root.mainloop()
