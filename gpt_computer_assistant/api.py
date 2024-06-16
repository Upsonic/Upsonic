# Create a python api and start_api function via flask

from flask import Flask, request, jsonify
import os
import sys
import threading
import time

from werkzeug.serving import make_server

app = Flask(__name__)

@app.route("/input", methods=["POST"])
def input():
    """
    This function receives input from the user and returns the response.
    """
    data = request.json
    text = data["text"]
    screen = data["screen"]
    print("Input:", text)
    from .gpt_computer_assistant import the_main_window, the_input_box
    if screen != "true":
        the_main_window.button_handler.input_text(text)
    else:
        the_main_window.button_handler.input_text_screenshot(text)
    
    time.sleep(1)
    while the_main_window.state != "idle":
        time.sleep(0.3)

    response = the_input_box.toPlainText()

    return jsonify({"response": response})



class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        threading.Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print("Starting server")
        self.srv.serve_forever()

    def shutdown(self):
        print("Stopping server")
        self.srv.shutdown()

server_thread = None

def start_api():
    global server_thread
    if server_thread is None:
        server_thread = ServerThread(app, "0.0.0.0", 7541)
        server_thread.start()
        print("API started")
    else:
        print("API is already running")

def stop_api():
    global server_thread
    if server_thread is not None:
        server_thread.shutdown()
        server_thread.join()
        server_thread = None
        print("API stopped")
    else:
        print("API is not running")