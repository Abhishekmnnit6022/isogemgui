"""Unified CAGD desktop application window."""

from __future__ import annotations

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QAction, QFileDialog,
    QMessageBox, QStatusBar, QStyleFactory
)
from PyQt5.QtCore import Qt

from core import io_json
from ui.bezier_tab import BezierTab
from ui.surface_tab import SurfaceTab
from ui.bspline_tab import BsplineTab


class MainWindow(QMainWindow):
    """Main window hosting Bezier Curve, Bezier Surface, and B-Spline tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IGA Designer-Geometric Modelling Platform for Isogeometric Analysis")
        self.resize(1100, 750)

        # Status Bar at the bottom
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self.set_status("Ready")

        # Central tab widget
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._bezier_tab = BezierTab(status_callback=self.set_status)
        self._surface_tab = SurfaceTab(status_callback=self.set_status)
        self._bspline_tab = BsplineTab(status_callback=self.set_status)

        self._tabs.addTab(self._bezier_tab, "Bezier Curve")
        self._tabs.addTab(self._surface_tab, "Bezier Surface")
        self._tabs.addTab(self._bspline_tab, "B-Spline")

        self._create_menu_bar()

    # Update the text in the status bar
    def set_status(self, message: str):
        self._status_bar.showMessage(message)

    # Set up File and Help menus
    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._file_new)
        file_menu.addAction(new_act)

        open_act = QAction("&Open...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._file_open)
        file_menu.addAction(open_act)

        save_act = QAction("&Save As...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._file_save_as)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        excel_act = QAction("&Export basis matrix to Excel...", self)
        excel_act.triggered.connect(self._export_excel)
        file_menu.addAction(excel_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._help_about)
        help_menu.addAction(about_act)

    # Clear contents of the active tab
    def _file_new(self):
        curr_idx = self._tabs.currentIndex()
        if curr_idx == 0:
            self._bezier_tab._clear_points()
            self.set_status("Cleared Bezier Curve data")
        elif curr_idx == 1:
            self._surface_tab._clear_coords()
            self.set_status("Cleared Bezier Surface coordinates")
        elif curr_idx == 2:
            self._bspline_tab._clear_points()
            self.set_status("Cleared B-Spline control points")

    # Open JSON file and load into the appropriate tab
    def _file_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            data = io_json.load(path)
            geom_type = data.get("type")
            if geom_type == "bezier_curve":
                self._tabs.setCurrentWidget(self._bezier_tab)
                self._bezier_tab.set_data(data)
                self.set_status(f"Loaded Bezier curve from {Path(path).name}")
            elif geom_type == "bezier_surface":
                self._tabs.setCurrentWidget(self._surface_tab)
                self._surface_tab.set_data(data)
                self.set_status(f"Loaded Bezier surface from {Path(path).name}")
            elif geom_type == "bspline_curve":
                self._tabs.setCurrentWidget(self._bspline_tab)
                self._bspline_tab.set_data(data)
                self.set_status(f"Loaded B-spline curve from {Path(path).name}")
            else:
                QMessageBox.warning(self, "Open File", f"Unrecognized geometry type: {geom_type}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}")

    # Save active tab data to a JSON file
    def _file_save_as(self):
        curr_tab = self._tabs.currentWidget()
        try:
            data = curr_tab.get_data()
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Cannot prepare data to save:\n{e}")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Geometry", f"{data['type']}.json", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            io_json.save(path, data)
            self.set_status(f"Saved {data['type']} to {Path(path).name}")
            QMessageBox.information(self, "Save Successful", f"Saved successfully to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")

    # Switch to B-spline tab and trigger Excel export
    def _export_excel(self):
        self._tabs.setCurrentWidget(self._bspline_tab)
        self._bspline_tab.export_basis_excel()

    # Show application about dialog
    def _help_about(self):
        QMessageBox.about(
            self,
            "About IGA Designer",
            "IGA Designer-Geometric Modelling Platform for Isogeometric Analysis\n\n"
            "Features:\n"
            "  * Bezier Curves (Arbitrary degree, 2D and 3D)\n"
            "  * Bezier Surfaces (Tensor product control nets, 3D interactive view)\n"
            "  * B-Spline Curves (Cox-de Boor recursion, arbitrary knots and degrees)\n"
            "  * Excel basis matrix export\n"
            "  * JSON geometry save/load\n\n"
            "Unified Architecture and Pure-Math Core."
        )


CAGDApp = MainWindow


# Apply classic Windows style
def apply_classic_style(app: QApplication):
    windows_style = QStyleFactory.create("Windows")
    if windows_style:
        app.setStyle(windows_style)


def main():
    app = QApplication(sys.argv)
    apply_classic_style(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
