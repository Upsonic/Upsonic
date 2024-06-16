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

    firsst_text = the_input_box.toPlainText()

    if screen != "true":
        the_main_window.button_handler.input_text(text)
    else:
        the_main_window.button_handler.input_text_screenshot(text)
    
    while the_input_box.toPlainText() == firsst_text:
        time.sleep(0.3)

    while the_input_box.toPlainText().startswith("System:"):
        time.sleep(0.3)

    response = the_input_box.toPlainText()

    return jsonify({"response": response})



@app.route("/profile", methods=["POST"])
def profile():
    """
    This function sets the profile for the application.
    """
    data = request.json
    profile = data["profile"]
    print("Profile:", profile)
    from .utils.db import set_profile
    set_profile(profile)
    from .gpt_computer_assistant import the_main_window
    the_main_window.update_from_thread("Profile set to "+profile)
    return jsonify({"response": "Profile set to "+profile})


@app.route("/reset_memory", methods=["POST"])
def reset_memory():
    """
    This function resets the memory of the application.
    """
    from .agent.chat_history import clear_chat_history
    clear_chat_history()
    from .gpt_computer_assistant import the_main_window
    the_main_window.update_from_thread("Memory reset")
    return jsonify({"response": "Memory reset"})



@app.route("/activate_predefined_agents", methods=["POST"])
def enable_predefined_agents():
    """
    This function enables predefined agents for the application.
    """
    from .utils.db import activate_predefined_agents_setting
    activate_predefined_agents_setting()
    from .gpt_computer_assistant import the_main_window
    the_main_window.update_from_thread("Predefined agents enabled")
    return jsonify({"response": "Predefined agents enabled"})

@app.route("/deactivate_predefined_agents", methods=["POST"])
def disable_predefined_agents():
    """
    This function disables predefined agents for the application.
    """
    from .utils.db import deactivate_predefined_agents_setting
    deactivate_predefined_agents_setting()
    from .gpt_computer_assistant import the_main_window
    the_main_window.update_from_thread("Predefined agents disabled")
    return jsonify({"response": "Predefined agents disabled"})


@app.route("/activate_online_tools", methods=["POST"])
def enable_online_tools():
    """
    This function enables online tools for the application.
    """
    from .utils.db import activate_online_tools_setting
    activate_online_tools_setting()
    from .gpt_computer_assistant import the_main_window
    the_main_window.update_from_thread("Online tools enabled")
    return jsonify({"response": "Online tools enabled"})



@app.route("/deactivate_online_tools", methods=["POST"])
def disable_online_tools():
    """
    This function disables online tools for the application.
    """
    from .utils.db import deactivate_online_tools_setting
    deactivate_online_tools_setting()
    from .gpt_computer_assistant import the_main_window
    the_main_window.update_from_thread("Online tools disabled")
    return jsonify({"response": "Online tools disabled"})


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