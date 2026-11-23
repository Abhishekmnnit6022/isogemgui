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
    """Replace QMessageBox pop-ups with a recorder so tests never block.

    Returns a list of (kind, title, text) tuples, one per message box shown.
    """
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
