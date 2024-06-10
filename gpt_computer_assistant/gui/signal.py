from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject



class SignalHandler(QObject):
    """
    A QObject subclass to handle signals used in the GUI application.

    This class defines several signals that can be used to communicate
    between different components of the GUI application.

    Signals:
    - recording_started: Signal emitted when recording is started.
    - recording_stopped: Signal emitted when recording is stopped.
    - assistant_thinking: Signal emitted when the assistant is processing a request.
    - assistant_response_ready: Signal emitted when the assistant response is ready to be displayed.
    - assistant_response_stopped: Signal emitted when the assistant response display is stopped.

    """
    
    def emit_connection(self, signal):
        recording_started = signal()
        recording_stopped = signal()
        assistant_thinking = signal()
        assistant_response_ready = signal()
        assistant_response_stopped = signal()
    
    try: 
        emit_connection(pyqtSignal)
    except Exception as e:
        print("Error in SignalHandler", e)
        print("Retrying in 5 seconds to avoid overloading requests query...")
        from time import sleep
        sleep(5)
        emit_connection(pyqtSignal)
        

signal_handler = SignalHandler()

