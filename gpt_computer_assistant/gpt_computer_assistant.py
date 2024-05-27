from .agent.chat_history import *
from .agent.assistant import *
from .llm import *
from .agent.agent import *
from .agent.background import *

from .gui.signal import *
from .gui.button import *

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
from PyQt5.QtGui import QMouseEvent, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
import os
import scipy.io.wavfile as wavfile

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QPoint










print("Imported all libraries")















class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # Remove title bar and set window to always on top
        
        self.state = 'idle'  # Possible states: 'idle', 'talking', 'thinking'
        self.pulse_timer = None
        self.initUI()
        self.button_handler = ButtonHandler(self)  # Pass the whole MainWindow object
        self.old_position = self.pos()  # For moving window

    def initUI(self):
        self.setWindowTitle('GPT-4o Assistant')
        self.setGeometry(100, 100, 200, 135)  # Adjust the size as needed
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Custom title bar
        self.title_bar = QWidget(self)
        self.title_bar.setFixedHeight(20)  # Set a fixed height for the title bar
        self.title_bar_layout = QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.title_bar_layout.setSpacing(0)

        self.title_label = QLabel("GPT-4o Assistant", self)
        self.title_bar_layout.addWidget(self.title_label)

        self.close_button = QPushButton("X", self)
        self.close_button.setFixedSize(20, 20)
        self.close_button.clicked.connect(self.close)
        self.title_bar_layout.addWidget(self.close_button)


        layout.addWidget(self.title_bar)

        # Add other UI elements below the title bar
        self.other_widget = QWidget(self)
        layout.addWidget(self.other_widget)

        self.setLayout(layout)

        # Add keyboard shortcuts
        self.shortcut_screenshot = QShortcut(QKeySequence("Ctrl+1"), self)
        self.shortcut_screenshot.activated.connect(lambda: self.button_handler.just_screenshot())        
        self.shortcut_screenshot = QShortcut(QKeySequence("Ctrl+2"), self)
        self.shortcut_screenshot.activated.connect(lambda: self.button_handler.toggle_recording(take_system_audio=True))

        self.shortcut_no_screenshot = QShortcut(QKeySequence("Ctrl+e"), self)
        self.shortcut_no_screenshot.activated.connect(lambda: self.button_handler.toggle_recording(take_system_audio=True))

        self.shortcut_no_screenshot = QShortcut(QKeySequence("Ctrl+3"), self)
        self.shortcut_no_screenshot.activated.connect(lambda: self.button_handler.toggle_recording(no_screenshot=True))


        self.show()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.old_position = event.globalPos()

    def mouseMoveEvent(self, event: QMouseEvent):
        delta = QPoint(event.globalPos() - self.old_position)
        if event.buttons() == Qt.LeftButton and self.title_bar.underMouse():
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_position = event.globalPos()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable antialiasing for smooth edges
        painter.setPen(QPen(Qt.black, 8, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))

        center_x = 100  # Fixed center x-coordinate
        center_y = 90  # Fixed center y-coordinate

        if self.state == 'talking':
            # Draw a pulsating circle with smooth easing animation
            radius_variation = 5 * (1 + math.sin(self.pulse_frame * math.pi / 100))
            radius = 70 + radius_variation
            painter.drawEllipse(int(center_x - radius / 2), int(center_y - radius / 2), int(radius), int(radius))
        elif self.state == 'thinking':
            # more slow pulsating circle with smooth easing animation
            radius_variation = 5 * (1 + math.sin(self.pulse_frame * math.pi / 100))
            radius = 70 + radius_variation
            painter.drawEllipse(int(center_x - radius / 2), int(center_y - radius / 2), int(radius), int(radius))

        else:
            radius = 70
            painter.drawEllipse(int(center_x - radius / 2), int(center_y - radius / 2), int(radius), int(radius))
        
        self.circle_rect = QRect(int(center_x - radius / 2), int(center_y - radius / 2), int(radius), int(radius))

        # Draw second smaller circle button at bottom right
        small_center_x = self.width() - 30
        small_center_y = self.height() - 30
        small_radius = 25
        painter.drawEllipse(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))
        
        self.small_circle_rect = QRect(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))



        # Draw second smaller circle button at bottom left
        small_center_x = 30  # Adjust this line
        small_center_y = self.height() - 30
        small_radius = 25
        painter.drawEllipse(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))

        self.small_circle_left = QRect(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))


        # Draw second smaller circle button at top left
        small_center_x = 30  # Adjust this line
        small_center_y = 60  # Adjusted this line
        small_radius = 25
        painter.drawEllipse(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))

        self.small_circle_left_top = QRect(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))


        # Draw second smaller rectangle button at top right
        small_center_x = self.width() - 30  # Adjust this line
        small_center_y = 60  # Adjusted this line
        
        painter.drawRect(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))

        self.small_rect_right_top = QRect(int(small_center_x - small_radius / 2), int(small_center_y - small_radius / 2), int(small_radius), int(small_radius))
        



    def update_state(self, new_state):
        self.state = new_state
        print(f"State updated: {new_state}")
        if new_state == 'talking':
            self.pulse_frame = 0
            if self.pulse_timer:
                self.pulse_timer.stop()
                self.pulse_timer = None
            self.pulse_timer = QTimer(self)
            self.pulse_timer.timeout.connect(self.pulse_circle)
            self.pulse_timer.start(10)
        elif new_state == 'thinking':
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
        self.pulse_frame += 1
        if self.pulse_frame >= 100:
            self.pulse_frame = 0
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.circle_rect.contains(event.pos()):
            self.button_handler.toggle_recording()
        elif self.small_circle_rect.contains(event.pos()):
            self.button_handler.toggle_recording(no_screenshot=True)
        elif self.small_circle_left.contains(event.pos()):
            self.button_handler.toggle_recording(take_system_audio=True)
        elif self.small_circle_left_top.contains(event.pos()):
            self.button_handler.just_screenshot()

        elif self.small_rect_right_top.contains(event.pos()):
            self.button_handler.settings_popup()

            

















def start():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
