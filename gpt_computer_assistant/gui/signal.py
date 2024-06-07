from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject


class SignalHandler(QObject):
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    assistant_thinking = pyqtSignal()
    assistant_response_ready = pyqtSignal()
    assistant_response_stopped = pyqtSignal()


signal_handler = SignalHandler()
