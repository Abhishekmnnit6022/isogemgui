"""Application launcher entry point for CAGD Suite."""

import sys
from PyQt5.QtWidgets import QApplication
from app import MainWindow, apply_classic_style


def main():
    app = QApplication(sys.argv)
    apply_classic_style(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
