try:
    from .agent.chat_history import *
    from .agent.assistant import *
    from .llm import *
    from .agent.agent import *
    from .agent.background import *

    from .gui.signal import *
    from .gui.button import *
    from .gui.settings import settings_popup
    from .gui.llmsettings import llmsettings_popup
    from .utils.db import load_api_key, load_model_settings, screenshot_icon_path, microphone_icon_path, audio_icon_path
    from .utils.telemetry import my_tracer, os_name

except ImportError:
    # This is for running the script directly
    # in order to test the GUI without rebuilding the package
    from agent.chat_history import *
    from agent.assistant import *
    from llm import *
    from agent.agent import *
    from agent.background import *
    from utils.db import load_api_key, load_model_settings, screenshot_icon_path, microphone_icon_path, audio_icon_path
    from gui.signal import *
    from gui.button import *
    from gui.settings import settings_popup
    from gui.llmsettings import llmsettings_popup
    from utils.telemetry import my_tracer, os_name


import hashlib
import sys
import threading
import base64
import time
import random
import numpy as np
import sounddevice as sd
import soundfile as sf

from pygame import mixer
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QMouseEvent, QPainter, QPen, QBrush, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
import os
import scipy.io.wavfile as wavfile

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QPoint

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QIcon

print("Imported all libraries")


from PyQt5 import QtCore


try:
    import ctypes

    myappid = "onuratakan.gpt_computer_assistant.gui.1"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except:
    pass

the_input_box = None


the_main_window = None


user_id = load_user_id()
os_name_ = os_name()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        self.state = "idle"
        self.pulse_timer = None

        self.button_handler = ButtonHandler(self)
        self.initUI()
        self.old_position = self.pos()

        if load_model_settings().startswith("gpt"):
            self.should_paint = True  # Flag to control painting
        else:
            self.should_paint = False




        

        global the_main_window
        the_main_window = self

    def initUI(self):
        self.setWindowTitle("GPT")
        self.setGeometry(100, 100, 200, 200)
        self.setFixedSize(self.width(), self.height() + 20)

        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_16_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_24_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_32_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_48_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_256_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Custom title bar
        self.title_bar = QWidget(self)
        self.title_bar.setFixedHeight(20)  # Set a fixed height for the title bar
        self.title_bar_layout = QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.title_bar_layout.setSpacing(0)

        layout.addWidget(self.title_bar)

        # Add other UI elements below the title bar
        self.other_widget = QWidget(self)
        layout.addWidget(self.other_widget)

        self.layout = layout

        self.setLayout(layout)

        # Add keyboard shortcuts
        self.shortcut_screenshot = QShortcut(QKeySequence("Ctrl+1"), self)
        self.shortcut_screenshot.activated.connect(
            lambda: self.button_handler.just_screenshot()
        )
        self.shortcut_screenshot = QShortcut(QKeySequence("Ctrl+2"), self)
        self.shortcut_screenshot.activated.connect(
            lambda: self.button_handler.toggle_recording(take_system_audio=True)
        )

        self.shortcut_no_screenshot = QShortcut(QKeySequence("Ctrl+e"), self)
        self.shortcut_no_screenshot.activated.connect(
            lambda: self.button_handler.toggle_recording(take_system_audio=True)
        )

        self.shortcut_no_screenshot = QShortcut(QKeySequence("Ctrl+3"), self)
        self.shortcut_no_screenshot.activated.connect(
            lambda: self.button_handler.toggle_recording(no_screenshot=True)
        )

        # I want to create an input box to bottom left and a send button to bottom right

        input_box = QLineEdit(self)

        if load_api_key() == "CHANGE_ME":
            input_box.setPlaceholderText("Save your API Key, go to settings")
        else:
            input_box.setPlaceholderText("Type here")
        input_box.setGeometry(30, self.height() - 60, 200, 30)
        global the_input_box
        the_input_box = input_box

        def input_box_send():
            if input_box.text() != "":
                self.button_handler.input_text(input_box.text())

        def input_box_send_screenshot():
            if input_box.text() != "":
                self.button_handler.input_text_screenshot(input_box.text())

        self.layout.addWidget(input_box)

        # Create a horizontal layout
        button_layout = QHBoxLayout()

        # Create the send button
        send_button = QPushButton("Send", self)
        send_button.clicked.connect(input_box_send)

        # Create the screenshot button
        self.screenshot_button = QPushButton("+Screenshot", self)
        self.screenshot_button.clicked.connect(input_box_send_screenshot)


        if load_model_settings() == "mixtral-8x7b-groq":
            self.screenshot_button.hide()



        # Add the buttons to the horizontal layout
        button_layout.addWidget(send_button)
        button_layout.addWidget(self.screenshot_button)

        self.shortcut_enter = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_enter.activated.connect(input_box_send_screenshot)
        self.shortcut_enter = QShortcut(QKeySequence("Return"), self)
        self.shortcut_enter.activated.connect(input_box_send)

        self.layout.addLayout(button_layout)

        button_layout_ = QHBoxLayout()

        settingsButton = QPushButton("Chat Settings", self)
        settingsButton.clicked.connect(settings_popup)

        llmsettingsButton = QPushButton("LLM Settings", self)
        llmsettingsButton.clicked.connect(llmsettings_popup)

        button_layout_.addWidget(settingsButton)
        button_layout_.addWidget(llmsettingsButton)
        self.layout.addLayout(button_layout_)

        self.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        delta = QPoint(event.globalPos() - self.old_position)
        if event.buttons() == Qt.LeftButton and self.title_bar.underMouse():
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_position = event.globalPos()

    def paintEvent(self, event):
        if not self.should_paint:
            return  # Skip the drawing if should_paint is False


        if load_model_settings() == "gpt-4o":
            self.screen_available = True
        else:
            self.screen_available = False

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 8, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))

        center_x = 100
        center_y = 50

        if self.state == "talking":
            # Draw a pulsating circle with smooth easing animation
            radius_variation = 5 * (1 + math.sin(self.pulse_frame * math.pi / 100))
            radius = 70 + radius_variation
            painter.drawEllipse(
                int(center_x - radius / 2),
                int(center_y - radius / 2),
                int(radius),
                int(radius),
            )
        elif self.state == "thinking":
            # more slow pulsating circle with smooth easing animation
            radius_variation = 5 * (1 + math.sin(self.pulse_frame * math.pi / 100))
            radius = 70 + radius_variation
            painter.drawEllipse(
                int(center_x - radius / 2),
                int(center_y - radius / 2),
                int(radius),
                int(radius),
            )

        else:
            radius = 70
            painter.drawEllipse(
                int(center_x - radius / 2),
                int(center_y - radius / 2),
                int(radius),
                int(radius),
            )

        self.circle_rect = QRect(
            int(center_x - radius / 2),
            int(center_y - radius / 2),
            int(radius),
            int(radius),
        )

        if self.screen_available:

            small_center_x = 170
            small_center_y = 25
            small_radius = 30
            painter.drawEllipse(
                int(small_center_x - small_radius / 2),
                int(small_center_y - small_radius / 2),
                int(small_radius),
                int(small_radius),
            )

            self.small_circle_rect = QRect(
                int(small_center_x - small_radius / 2),
                int(small_center_y - small_radius / 2),
                int(small_radius),
                int(small_radius),
            )

            # Draw the icon inside the circle
            icon_size = small_radius * 2 // 3  # Adjust the icon size relative to the circle
            icon_rect = QRect(
                small_center_x - icon_size // 2,
                small_center_y - icon_size // 2,
                icon_size,
                icon_size,
            )
            self.small_circle_recticon = QIcon(microphone_icon_path)
            self.small_circle_recticon.paint(painter, icon_rect)

            
            small_center_x = 30
            small_center_y = 65
            small_radius = 30
            painter.drawEllipse(
                int(small_center_x - small_radius / 2),
                int(small_center_y - small_radius / 2),
                int(small_radius),
                int(small_radius),
            )

            self.small_circle_left = QRect(
                int(small_center_x - small_radius / 2),
                int(small_center_y - small_radius / 2),
                int(small_radius),
                int(small_radius),
            )

            # Draw the icon inside the circle
            icon_size = small_radius * 2 // 3  # Adjust the icon size relative to the circle
            icon_rect = QRect(
                small_center_x - icon_size // 2,
                small_center_y - icon_size // 2,
                icon_size,
                icon_size,
            )
            self.small_circle_lefticon = QIcon(audio_icon_path)
            self.small_circle_lefticon.paint(painter, icon_rect)



            small_center_x = 30
            small_center_y = 25
            small_radius = 30
            painter.drawEllipse(
                int(small_center_x - small_radius / 2),
                int(small_center_y - small_radius / 2),
                int(small_radius),
                int(small_radius),
            )

            self.small_circle_left_top = QRect(
                int(small_center_x - small_radius / 2),
                int(small_center_y - small_radius / 2),
                int(small_radius),
                int(small_radius),
            )

            # Draw the icon inside the circle
            icon_size = small_radius * 2 // 3  # Adjust the icon size relative to the circle
            icon_rect = QRect(
                small_center_x - icon_size // 2,
                small_center_y - icon_size // 2,
                icon_size,
                icon_size,
            )
            self.small_circle_left_topticon = QIcon(screenshot_icon_path)
            self.small_circle_left_topticon.paint(painter, icon_rect)






    def remove_painting(self):
        self.should_paint = False  # Set the flag to False
        self.update()  # Request a repaint, which will now skip drawing

    def activate_painting(self):
        self.should_paint = True
        self.update()

    def remove_screenshot_button(self):
        self.screenshot_button.hide()

    def add_screenshot_button(self):
        self.screenshot_button.show()

    def update_state(self, new_state):
        self.state = new_state
        print(f"State updated: {new_state}")
        if new_state == "talking":
            self.pulse_frame = 0
            if self.pulse_timer:
                self.pulse_timer.stop()
                self.pulse_timer = None
            self.pulse_timer = QTimer(self)
            self.pulse_timer.timeout.connect(self.pulse_circle)
            self.pulse_timer.start(5)
        elif new_state == "thinking":
            the_input_box.setText("Thinking...")
            self.pulse_frame = 0
            if self.pulse_timer:
                self.pulse_timer.stop()
                self.pulse_timer = None
            self.pulse_timer = QTimer(self)
            self.pulse_timer.timeout.connect(self.pulse_circle)
            self.pulse_timer.start(20)
        elif self.pulse_timer:
            self.pulse_timer.stop()
            self.pulse_timer = None
        self.update()  # Trigger a repaint

    def pulse_circle(self):
        self.pulse_frame = (self.pulse_frame + 1) % 100
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        with my_tracer.start_span("mouse_press_event") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("os_name", os_name_)
            if self.state == "idle" or self.state == "talking":
                if self.circle_rect.contains(event.pos()):
                    if load_model_settings() == "gpt-4o":
                        
                        self.button_handler.toggle_recording(dont_save_image=True)
                    else:
                        self.button_handler.toggle_recording(no_screenshot=True)
                elif self.small_circle_rect.contains(event.pos()):
                    self.button_handler.toggle_recording(no_screenshot=True)
                elif self.small_circle_left.contains(event.pos()):
                    self.button_handler.toggle_recording(take_system_audio=True)
                elif self.small_circle_left_top.contains(event.pos()):
                    self.button_handler.just_screenshot()
