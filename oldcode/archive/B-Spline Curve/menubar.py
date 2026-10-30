import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        line_edit = QLineEdit(self)
        line_edit.setStyleSheet("color: black;")  # Set the text color to black

        self.setWindowTitle('LineEdit Example')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
