"""Shared pytest fixtures: a headless Qt application and captured message boxes."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for all GUI tests, running without a visible window."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes instead of showing them. Returns a list of (kind, title, text)."""
    from PyQt5.QtWidgets import QMessageBox
    shown = []

    def make(kind):
        def fake(parent, title, text, *args, **kwargs):
            shown.append((kind, title, text))
            return QMessageBox.Ok
        return fake

    for kind in ("warning", "information", "critical", "question", "about"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(make(kind)))
    return shown


@pytest.fixture
def window(qapp):
    """A fresh main window (all six tabs)."""
    from app import MainWindow
    return MainWindow()
