import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QSizePolicy

T_VALUE = 100


class BsplinesGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("B-Spline Basis Functions GUI")
        self.setStyleSheet(
            """
            QWidget {
                background-color:#e27e91;
                color: "white";
            }

            QLabel {
                font-size: 20px;
                color:"black";
            }

            QLineEdit {
                background-color: "black";
                font-size: 20px;
            }

            QPushButton {
                font-size: 20px;
                background-color: #4CAF50;
                color: white;
                border: dotted;
                padding: 10px 20px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: "red";
                font-size: 22px;
                cursor: pointer;
            }
            """
        )
        self.layout = QVBoxLayout()

        self.knots_label = QLabel("Knots (comma-separated):")
        # self.knots_label.setStyleSheet("font-size: 20px;") # To style Specific Label

        self.knots_input = QLineEdit()
        # self.knots_input.setStyleSheet("font-size: 20px;")
        self.knots_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.degree_label = QLabel("Degree:")

        self.degree_input = QLineEdit()
        self.degree_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.control_points_input = QLabel("Enter Control Points:")
        self.x_coordinate_label = QLabel("X-Coordinate:")
        self.x_coordinate_line_edit = QLineEdit()

        self.y_coordinate_label = QLabel("Y-Coordinate:")
        self.y_coordinate_line_edit = QLineEdit()

        self.control_points_button = QPushButton("Submit Control Point")
        self.control_points_button.clicked.connect(self.submit_control_points)

        self.basis_button = QPushButton("Plot B-Spline Basis Functions")
        # self.basis_button.setStyleSheet("font-size: 20px;") # To style Specific Button
        self.basis_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.basis_button.clicked.connect(self.plot_bspline)

        # Adding Bspline Curve Part button
        self.bspline_curve_button = QPushButton("Draw B-Spline Curve")
        self.bspline_curve_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.bspline_curve_button.clicked.connect(self.draw_bspline_curve)

        # Adding Layout that how all widget are going to arrange on the interface
        self.layout.addWidget(self.knots_label)
        self.layout.addWidget(self.knots_input)
        self.layout.addWidget(self.degree_label)
        self.layout.addWidget(self.degree_input)
        self.layout.addWidget(self.control_points_input)
        self.layout.addWidget(self.x_coordinate_label)
        self.layout.addWidget(self.x_coordinate_line_edit)
        self.layout.addWidget(self.y_coordinate_label)
        self.layout.addWidget(self.y_coordinate_line_edit)
        self.layout.addWidget(self.control_points_button)
        self.layout.addWidget(self.basis_button)
        self.layout.addWidget(self.bspline_curve_button)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(30, 30, 30, 30)

        self.setLayout(self.layout)

    def bspline_basis(self, i, p, t, knots):
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

        # Recursive call of bspline_basis
        return coeff1 * self.bspline_basis(i, p - 1, t, knots) + coeff2 * self.bspline_basis(i + 1, p - 1, t, knots)

    def plot_bspline(self):
        knots_input = self.knots_input.text().split(",")
        knots = [float(knot.strip()) for knot in knots_input]

        degree = int(self.degree_input.text())

        # Plotting code
        t = np.linspace(knots[0], knots[-1], T_VALUE, endpoint=True)
        basis = np.zeros((len(knots) - degree - 1, len(t)))

        for i in range(len(knots) - degree - 1):
            for j in range(len(t)):
                basis[i, j] = self.bspline_basis(i, degree, t[j], knots)

        plt.figure(figsize=(8, 6))
        for i in range(len(knots) - degree - 1):
            plt.plot(t, basis[i, :], label=f'N_{i}_{degree}')

        plt.legend(fontsize=12)
        plt.xlabel('t', fontsize=14)
        plt.ylabel('Basis value', fontsize=14)
        plt.title(f'B-Spline Basis Functions of degree {degree} with Knot Vector {knots}', fontsize=16)
        plt.show()

        return basis

    def submit_control_points(self):
        x_input = self.x_coordinate_line_edit.text()
        y_input = self.y_coordinate_line_edit.text()

        x_data = np.fromstring(x_input, sep=',')
        y_data = np.fromstring(y_input, sep=',')

        coordinate = np.array([x_data, y_data])
        print(coordinate)
        return coordinate

    def draw_bspline_curve(self):
        control_points = self.submit_control_points()
        basis = self.plot_bspline()
        degree = int(self.degree_input.text())
        t = np.linspace(0, 1, T_VALUE)
        curve_points = np.zeros(
            (control_points.shape[0], len(t)))  # shape[row i.e 0] 2 row and 100(len of t) column [2,100]
        # curve_points = np.zeros((2, len(t)))
        for i in range(len(t)):
            for j in range(control_points.shape[1]):  # shape[1] gives total no. fo column in control points array
                curve_points[:, i] += basis[j, i] * control_points[:, j]

        plt.figure()
        plt.plot(control_points[0, :], control_points[1, :], 'bo-', label='Control Points')
        plt.plot(curve_points[0, :], curve_points[1, :], 'r-', label='B-Spline Curve')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'B-Spline Curve having Degree {degree} and Order {degree + 1} with Control Points')
        plt.show()


if __name__ == '__main__':
    app = QApplication([])
    window = BsplinesGUI()
    window.show()
    app.exec()
