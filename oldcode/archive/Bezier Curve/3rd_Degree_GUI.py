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

    def third_degree():
        u = 0
        t_x = []
        t_y = []
        while u <= 1:
            put_x = pow(1 - u, 3) * list_y[0] + 3 * u * pow(1 - u, 2) * list_y[2] + 3 * u * u * pow(1 - u, 1) * \
                    list_y[4] \
                    + pow(u, 3) * list_y[6]
            put_y = pow(1 - u, 3) * list_y[1] + 3 * u * pow(1 - u, 2) * list_y[3] + 3 * u * u * pow(1 - u, 1) * list_y[5] \
                    + pow(u, 3) * list_y[7]
            t_x.append(put_x)
            t_y.append(put_y)
            u += 0.001
        # To separate x and y co-ordinates
        cod_x = []
        cod_y = []
        g=0
        h=1
        for t in range(len(list_y)-4):
            cod_x.append(list_y[g])
            cod_y.append(list_y[h])
            g+=2
            h+=2
        print(cod_x, cod_y)
        name = ["First Order Bezier", "Second Order Bezier", "n=3"]
        plt.title(name[order - 1])
        plt.plot(cod_x, cod_y,'o')
        plt.plot(t_x, t_y, color='g', label='curve')
        plt.show()

    if order == 3:
        for i in range(1, order + 2):
            cp_label = Label(text=f"Enter {i}st CP:", font=('Arial', 11, "bold"))
            cp_label.grid(column=0, row=i,padx=10, pady=10)

        for x in range(order + 1):
            for j in range(order - 1):
                cp_input = Entry(root,font=('Arial', 11))
                cp_input.grid(column=j + 1, row=x + 1)
                list_x.append(cp_input)

        ok_button1 = Button(text="ok", command=cp_input1)
        ok_button1.grid(column=3, row=2)

        clear = Button(root, text="Clear", command=clear_input)
        clear.grid(column=1, row=5)

        draw_button = Button(text="Draw", command=third_degree)
        draw_button.grid(column=2, row=5)


# GUI Part

root = Tk()
root.minsize(width=600, height=500)
root.title("Bezier Curve")
root.config(pady=20, padx=20)

# For order interface
order_label = Label(text="Enter Degree", font=('Arial', 11, "bold"))
order_label.grid(column=0, row=0)
order_input = Entry(root,font=('Arial', 11))
order_input.grid(column=1, row=0)
ok_button = Button(text="ok", command=input_order)
ok_button.grid(column=2, row=0)

root.mainloop()
