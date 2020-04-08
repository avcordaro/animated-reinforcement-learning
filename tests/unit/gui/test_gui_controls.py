import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from view.gui import GUI
from tkinter import *


@pytest.fixture(scope="module")
def gui():
    root = Tk()
    pytest.gui = GUI(root, None)
    return pytest.gui


def test_add_text_to_listbox(gui):
    str = "Test Information"
    pytest.gui.add_to_listbox(str)
    displayed_str = pytest.gui.listbox.get(END)
    assert displayed_str == str


def test_update_episode_labels(gui):
    new_episode_number = 10
    pytest.gui.update_episode_labels(new_episode_number)
    displayed_number = pytest.gui.episode_number.get()
    assert displayed_number == "Episode: 10"
