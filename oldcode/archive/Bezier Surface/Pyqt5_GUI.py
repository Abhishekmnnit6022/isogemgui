import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class BezierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bezier Surface Plot")
        self.setGeometry(100, 100, 600, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.layout = QVBoxLayout(central_widget)

        self.x_label = QLabel("X-coordinates as 'x1,x2,x3':", self)
        self.x_label.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.x_label)

        self.x_textbox = QLineEdit(self)
        self.x_textbox.setFont(QFont('Arial',11))
        self.layout.addWidget(self.x_textbox)

        self.y_label = QLabel("Y-coordinates as 'y1,y2,y3':", self)
        self.y_label.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.y_label)

        self.y_textbox = QLineEdit(self)
        self.y_textbox.setFont(QFont('Arial', 11))
        self.layout.addWidget(self.y_textbox)

        self.z_label = QLabel("Z-coordinates as 'z1,z2,z3':", self)
        self.z_label.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.z_label)

        self.z_textbox = QLineEdit(self)
        self.z_textbox.setFont(QFont('Arial', 11))
        self.layout.addWidget(self.z_textbox)

        self.plot_button = QPushButton("Plot Bezier Surface", self)
        self.plot_button.clicked.connect(self.plot_bezier_surface)
        self.layout.addWidget(self.plot_button)

        self.plot_widget = QLabel(self)
        self.plot_widget.setFixedSize(540, 400)
        self.layout.addWidget(self.plot_widget)

        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignTop)

    def plot_bezier_surface(self):
        x_input = self.x_textbox.text()
        y_input = self.y_textbox.text()
        z_input = self.z_textbox.text()

        x_coords = np.fromstring(x_input, sep=',')
        y_coords = np.fromstring(y_input, sep=',')
        z_coords = np.fromstring(z_input, sep=',')

        control_points = np.array([x_coords, y_coords, z_coords]).T

        x = control_points[:, 0]
        y = control_points[:, 1]
        z = control_points[:, 2]

        # Number of Cells for Each Direction
        uCELLS = 12
        wCELLS = 10

        # Dependent Variables
        uPTS = np.size(x, 0)
        wPTS = np.size(x, 1)
        n = uPTS - 1
        m = wPTS - 1

        # Parametric Variable
        u = np.linspace(0, 1, uCELLS)
        w = np.linspace(0, 1, wCELLS)

        # Bernstein Basis Polynomial
        b = []
        d = []

        xBezier = np.zeros((uCELLS, wCELLS))
        yBezier = np.zeros((uCELLS, wCELLS))
        zBezier = np.zeros((uCELLS, wCELLS))

        for i in range(0, uPTS):
            for j in range(0, wPTS):
                b.append(self.J(n, i, u))
                d.append(self.K(m, j, w))
                Jt = self.J(n, i, u).transpose()
                xBezier = Jt * self.K(m, j, w) * x[i, j] + xBezier
                yBezier = Jt * self.K(m, j, w) * y[i, j] + yBezier
                zBezier = Jt * self.K(m, j, w) * z[i, j] + zBezier

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xBezier, yBezier, zBezier)
        ax.scatter(x, y, z, edgecolors='face')

        # Display the plot in GUI
        plt.savefig('plot.png')  # Save plot to file
        self.plot_widget.setPixmap('plot.png')

    def J(self, n, i, u):
        return np.matrix(self.Ni(n, i) * (u ** i) * (1 - u) ** (n - i))

    def K(self, m, j, w):
        return np.matrix(self.Mj(m, j) * (w ** j) * (1 - w) ** (m - j))

    def Ni(self, n, i):
        return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))

    def Mj(self, m, j):
        return np.math.factorial(m) / (np.math.factorial(j) * np.math.factorial(m - j))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BezierGUI()
    window.show()
    sys.exit(app.exec_())
