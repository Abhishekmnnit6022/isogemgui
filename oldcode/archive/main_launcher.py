"""
CAGD Curve & Surface Studio - Unified Multi-Tool Launcher
Provides a centralized, modern interface to launch and manage all B-Spline,
Bézier Curve, and Bézier Surface applications.
"""

import os
import sys
import subprocess
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QScrollArea, QFrame,
    QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

class ToolCard(QFrame):
    def __init__(self, title, description, category_tag, script_path, is_console=False, on_launch_callback=None):
        super().__init__()
        self.title = title
        self.description = description
        self.category_tag = category_tag
        self.script_path = script_path
        self.is_console = is_console
        self.on_launch_callback = on_launch_callback

        self.setObjectName("ToolCard")
        self.setStyleSheet("""
            #ToolCard {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 12px;
                padding: 16px;
            }
            #ToolCard:hover {
                border: 1px solid #6366f1;
                background-color: #242438;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header: Title + Tag
        header_layout = QHBoxLayout()
        title_label = QLabel(self.title)
        title_label.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        tag_label = QLabel(self.category_tag)
        tag_label.setStyleSheet("""
            background-color: #312e81;
            color: #a5b4fc;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
        """)
        header_layout.addWidget(tag_label)
        layout.addLayout(header_layout)

        # Description
        desc_label = QLabel(self.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #94a3b8; font-size: 13px; line-height: 1.4;")
        layout.addWidget(desc_label)

        # Footer: Action button + file info
        footer_layout = QHBoxLayout()
        file_hint = QLabel(os.path.basename(self.script_path))
        file_hint.setStyleSheet("color: #64748b; font-size: 11px; font-family: monospace;")
        footer_layout.addWidget(file_hint)

        footer_layout.addStretch()

        launch_btn = QPushButton("Launch Tool ➜" if not is_console else "Open Terminal ➜")
        launch_btn.setCursor(Qt.PointingHandCursor)
        launch_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
            }
            QPushButton:pressed {
                background-color: #4338ca;
            }
        """)
        launch_btn.clicked.connect(self.launch)
        footer_layout.addWidget(launch_btn)

        layout.addLayout(footer_layout)

    def launch(self):
        full_path = os.path.join(BASE_DIR, self.script_path)
        if not os.path.exists(full_path):
            QMessageBox.critical(self, "File Not Found", f"Could not find script:\n{full_path}")
            return

        script_dir = os.path.dirname(full_path)
        try:
            if self.is_console:
                # Open interactive console window for scripts expecting terminal input
                cmd = f'start cmd.exe /k "{PYTHON_EXE} \\"{full_path}\\""'
                subprocess.Popen(cmd, cwd=script_dir, shell=True)
            else:
                subprocess.Popen([PYTHON_EXE, full_path], cwd=script_dir)

            if self.on_launch_callback:
                self.on_launch_callback(self.title, self.script_path)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to start {self.title}:\n{str(e)}")


class UnifiedLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAGD Geometric Modeling Suite - Master Launcher")
        self.resize(960, 720)
        self.setMinimumSize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        # Window dark background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QTabWidget::pane {
                border: 1px solid #1e293b;
                background-color: #0f172a;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #6366f1;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #334155;
                color: #f1f5f9;
            }
            QScrollBar:vertical {
                border: none;
                background: #0f172a;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QStatusBar {
                background-color: #0b0f19;
                color: #94a3b8;
                font-size: 12px;
                border-top: 1px solid #1e293b;
                padding: 4px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(28, 24, 28, 16)
        main_layout.setSpacing(16)

        # Top Banner
        banner_layout = QVBoxLayout()
        app_title = QLabel("📐 Computer-Aided Geometric Design Studio")
        app_title.setStyleSheet("color: #f8fafc; font-size: 24px; font-weight: 800; letter-spacing: -0.5px;")
        banner_layout.addWidget(app_title)

        subtitle = QLabel("Central control hub for B-Spline Curves, Bézier Curves, and 3D Bézier Surface modeling tools.")
        subtitle.setStyleSheet("color: #94a3b8; font-size: 14px;")
        banner_layout.addWidget(subtitle)
        main_layout.addLayout(banner_layout)

        # Tabs for categories
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_bspline_tab(), "🔷 B-Spline Curves")
        self.tabs.addTab(self.create_bezier_tab(), "🔶 Bézier Curves")
        self.tabs.addTab(self.create_surface_tab(), "🌐 Bézier Surfaces")

        main_layout.addWidget(self.tabs)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Ready. Running on Python {sys.version.split()[0]} ({PYTHON_EXE})")

    def create_tab_scroll_area(self, cards):
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(14)

        for card in cards:
            layout.addWidget(card)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        return scroll

    def create_bspline_tab(self):
        cards = [
            ToolCard(
                title="B-Spline Studio GUI (Full Suite)",
                description="Comprehensive PyQt5 GUI with menu bar, knot vector management, polynomial degree selection, control points submission, and live 2D B-spline curve visualization.",
                category_tag="PyQt5 Desktop App",
                script_path=os.path.join("B-Spline Curve", "practiceGUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="B-Spline Basis & Curve Plotter",
                description="Interactive PyQt5 interface for calculating Cox-de Boor basis functions and plotting resulting curves with user-defined control point matrices.",
                category_tag="PyQt5 Plotter",
                script_path=os.path.join("B-Spline Curve", "Pyqt5GUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="Tkinter B-Spline Basis Visualizer",
                description="Lightweight Tkinter window featuring live embedded Matplotlib canvas to observe basis function distribution across arbitrary knot vectors.",
                category_tag="Tkinter GUI",
                script_path=os.path.join("B-Spline Curve", "BsplineCurveGUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="Cox-de Boor Basis Function Generator",
                description="Step-by-step 0th and higher-order Cox-de Boor basis function evaluation script with sequential degree-by-degree graphical breakdown.",
                category_tag="Interactive CLI",
                script_path=os.path.join("B-Spline Curve", "Basis_function.py"),
                is_console=True,
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="Circular B-Spline Evaluator",
                description="Demonstrates closed circular B-spline curve constructions using symmetric knot vectors and periodic circular control points.",
                category_tag="CLI / Matplotlib",
                script_path=os.path.join("B-Spline Curve", "CircleBspline.py"),
                is_console=True,
                on_launch_callback=self.on_tool_launched
            )
        ]
        return self.create_tab_scroll_area(cards)

    def create_bezier_tab(self):
        cards = [
            ToolCard(
                title="General Bézier Curve GUI",
                description="Interactive point-and-click tool to input arbitrary 2D control points and generate high-degree Bézier curves using Bernstein polynomials.",
                category_tag="Tkinter Interactive",
                script_path=os.path.join("Bezier Curve", "General_Bezier_Curve.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="3D Cubic Bézier Curve Plotter",
                description="Generates and visualizes 3-dimensional cubic Bézier curves through 3D space with 4 control points.",
                category_tag="3D Matplotlib GUI",
                script_path=os.path.join("Bezier Curve", "3D_Cubic_GUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="3rd Degree (Cubic) Bézier GUI",
                description="Dedicated cubic Bézier curve evaluator with standard 4 control point configuration.",
                category_tag="Tkinter GUI",
                script_path=os.path.join("Bezier Curve", "3rd_Degree_GUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="2nd Degree (Quadratic) Bézier GUI",
                description="Interactive quadratic Bézier curve generator using 3 control points.",
                category_tag="Tkinter GUI",
                script_path=os.path.join("Bezier Curve", "2nd_Degree_GUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="1st Degree (Linear) Bézier GUI",
                description="Linear Bézier segment visualization demonstrating fundamental linear interpolation between 2 endpoints.",
                category_tag="Tkinter GUI",
                script_path=os.path.join("Bezier Curve", "1st_Degree_GUI.py"),
                on_launch_callback=self.on_tool_launched
            )
        ]
        return self.create_tab_scroll_area(cards)

    def create_surface_tab(self):
        cards = [
            ToolCard(
                title="3D Bézier Surface GUI",
                description="PyQt5 application for defining 2D control nets (X, Y, Z coordinate matrices) and rendering 3D Bézier tensor-product surfaces.",
                category_tag="PyQt5 3D Surface",
                script_path=os.path.join("Bezier Surface", "Pyqt5_GUI.py"),
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="3D Surface Convex Hull Validator",
                description="Analyzes the spatial convex hull property for control grids and tests bounding volumes of Bézier surface patches.",
                category_tag="CLI / Analysis",
                script_path=os.path.join("Bezier Surface", "Convex_Hull_Check.py"),
                is_console=True,
                on_launch_callback=self.on_tool_launched
            ),
            ToolCard(
                title="Command-Based Bézier Surface Engine",
                description="Command-line driven surface parameterization script evaluating tensor-product Bernstein basis functions.",
                category_tag="CLI Engine",
                script_path=os.path.join("Bezier Surface", "Command_Based.py"),
                is_console=True,
                on_launch_callback=self.on_tool_launched
            )
        ]
        return self.create_tab_scroll_area(cards)

    def on_tool_launched(self, title, script_path):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_bar.showMessage(f"[{timestamp}] Successfully launched: {title} ({script_path})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    launcher = UnifiedLauncher()
    launcher.show()
    sys.exit(app.exec_())
