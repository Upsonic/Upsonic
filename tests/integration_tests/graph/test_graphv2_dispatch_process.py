"""Real process death after an independent provider SQLite commit."""
import json
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

CHILD = r'''
import json, sqlite3, sys, time
from typing_extensions import TypedDict
from upsonic.graphv2 import StateGraph, SqliteCheckpointer, START, END
mode, path, durability, topology = sys.argv[1:]
class State(TypedDict):
    receipt: str
def prepare(state):
    return {}
def pay(state, config):
    with sqlite3.connect(path + '/provider.db') as provider:
        provider.execute('CREATE TABLE IF NOT EXISTS effects (id INTEGER PRIMARY KEY, dispatch_id TEXT)')
        provider.execute('INSERT INTO effects (dispatch_id) VALUES (?)',
                         (config.get('configurable', {}).get('node_execution_id'),))
    if mode == 'start':
        while True: time.sleep(.1)
    return {'receipt': 'paid'}
builder = StateGraph(State)
builder.add_node('pay', pay)
if topology == 'prepare':
    builder.add_node('prepare', prepare)
    builder.add_edge(START, 'prepare')
    builder.add_edge('prepare', 'pay')
else:
    builder.add_edge(START, 'pay')
builder.add_edge('pay', END)
connection = sqlite3.connect(path + '/graph.db', check_same_thread=False)
graph = builder.compile(checkpointer=SqliteCheckpointer(connection), durability=durability)
try:
    result = graph.invoke({'receipt': ''} if mode == 'start' else None,
                          config={'configurable': {'thread_id': 'invoice-1'}})
    print(json.dumps({'result': result}), flush=True)
except Exception as error:
    checkpoint = graph.checkpointer.get('invoice-1')
    print(json.dumps({'error': type(error).__name__, 'message': str(error),
                      'pending_tasks': getattr(checkpoint, 'pending_tasks', None)}), flush=True)
finally:
    connection.close()
'''

def count_effects(path):
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path) as connection:
            return connection.execute('SELECT COUNT(*) FROM effects').fetchone()[0]
    except sqlite3.OperationalError:
        return 0

@pytest.mark.parametrize('durability', ['sync', 'async', 'exit'])
@pytest.mark.parametrize('topology', ['first', 'prepare'])
def test_process_death_blocks_unsafe_reexecution(tmp_path, durability, topology):
    args = [sys.executable, '-c', CHILD]
    process = subprocess.Popen(args + ['start', str(tmp_path), durability, topology],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while count_effects(tmp_path / 'provider.db') == 0 and process.poll() is None:
            assert time.monotonic() < deadline, 'provider did not commit within 30 seconds'
            time.sleep(.025)
        assert count_effects(tmp_path / 'provider.db') == 1
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
    resumed = subprocess.run(args + ['resume', str(tmp_path), durability, topology],
                             capture_output=True, text=True, check=False, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    outcome = json.loads(resumed.stdout.strip().splitlines()[-1])
    assert count_effects(tmp_path / 'provider.db') == 1, outcome
    assert outcome.get('error') == 'GraphExecutionUncertainError', outcome
    with sqlite3.connect(tmp_path / 'provider.db') as connection:
        dispatch_id = connection.execute('SELECT dispatch_id FROM effects').fetchone()[0]
    assert dispatch_id, 'provider commit did not receive a logical dispatch identity'
    task = next(t for t in outcome['pending_tasks'] if t['node'] == 'pay')
    assert task['id'] == dispatch_id
    assert task['status'] in ('DISPATCH_RECORDED', 'UNKNOWN')


@pytest.mark.parametrize('parallel', [False, True])
def test_http_commit_with_reply_in_flight_blocks_new_process_resume(tmp_path, parallel):
    """Same lost-ack window as recovered repro, with a dynamically chosen port."""
    effects = []
    committed = threading.Event()
    release_response = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            effects.append(body)
            committed.set()
            release_response.wait(30)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"receipt":"paid"}')
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    # Replace only the provider operation; graph construction and fresh-process
    # recovery use the same API as the independent SQLite fixture.
    start = CHILD.index('def pay(state, config):')
    end = CHILD.index("builder = StateGraph(State)")
    http_body = f'''def pay(state, config):
    import urllib.request
    payload = json.dumps({{'dispatch_id': config.get('configurable', {{}}).get('node_execution_id')}}).encode()
    request = urllib.request.Request('http://127.0.0.1:{server.server_port}/pay', data=payload, method='POST')
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())
'''
    child = CHILD[:start] + http_body + CHILD[end:]
    if parallel:
        good = '''def good(state, config):
    with sqlite3.connect(path + '/sibling.db') as provider:
        provider.execute('CREATE TABLE IF NOT EXISTS effects (id INTEGER PRIMARY KEY, dispatch_id TEXT)')
        provider.execute('INSERT INTO effects (dispatch_id) VALUES (?)',
                         (config['configurable']['node_execution_id'],))
    return {'receipt': 'sibling-completed'}
'''
        child = child.replace('builder = StateGraph(State)', good + 'builder = StateGraph(State)')
        begin = child.index("builder.add_node('pay', pay)")
        finish = child.index('connection = sqlite3.connect')
        child = child[:begin] + '''builder.add_node('good', good)
builder.add_node('pay', pay)
builder.add_edge(START, 'good')
builder.add_edge(START, 'pay')
builder.add_edge('good', END)
builder.add_edge('pay', END)
''' + child[finish:]
    args = [sys.executable, '-c', child]
    process = subprocess.Popen(args + ['start', str(tmp_path), 'sync', 'prepare'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert committed.wait(20), 'HTTP provider did not commit'
        if parallel:
            from upsonic.graphv2 import SqliteCheckpointer
            with sqlite3.connect(tmp_path / 'graph.db') as connection:
                checkpoint = SqliteCheckpointer(connection).get('invoice-1')
            good_task = next(t for t in checkpoint.pending_tasks if t['node'] == 'good')
            assert good_task['status'] == 'COMPLETED'
            assert good_task['result'] == {'receipt': 'sibling-completed'}
        process.kill()
        process.communicate(timeout=10)
        # The process has died while its committed response remains withheld.
        resumed = subprocess.run(args + ['resume', str(tmp_path), 'sync', 'prepare'],
                                 capture_output=True, text=True, check=False, timeout=25)
        assert resumed.returncode == 0, resumed.stderr
        outcome = json.loads(resumed.stdout.strip().splitlines()[-1])
        assert len(effects) == 1, outcome
        assert outcome.get('error') == 'GraphExecutionUncertainError', outcome
        task = next(t for t in outcome['pending_tasks'] if t['node'] == 'pay')
        assert effects[0]['dispatch_id']
        assert task['id'] == effects[0]['dispatch_id']
        assert task['status'] in ('DISPATCH_RECORDED', 'UNKNOWN')
        if parallel:
            assert count_effects(tmp_path / 'sibling.db') == 1
            recovered_good = next(t for t in outcome['pending_tasks'] if t['node'] == 'good')
            assert recovered_good == good_task
            assert recovered_good['id'] != task['id']
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        release_response.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)


@pytest.mark.parametrize('durability', ['sync', 'async', 'exit'])
@pytest.mark.parametrize('boundary', ['NOT_STARTED', 'DISPATCH_RECORDED'])
def test_process_death_at_persisted_dispatch_boundary(tmp_path, durability, boundary):
    """Persisted plan permits recovery; persisted dispatch requires uncertainty."""
    barrier = f'''class BarrierSaver(SqliteCheckpointer):
    def put(self, checkpoint):
        super().put(checkpoint)
        tasks = getattr(checkpoint, 'pending_tasks', None) or []
        if mode == 'start' and tasks and tasks[0]['status'] == {boundary!r}:
            with open(path + '/boundary.json', 'w') as marker:
                json.dump(tasks[0], marker)
            while True: time.sleep(.1)
'''
    child = CHILD.replace('connection = sqlite3.connect', barrier + 'connection = sqlite3.connect')
    child = child.replace('checkpointer=SqliteCheckpointer(connection)', 'checkpointer=BarrierSaver(connection)')
    args = [sys.executable, '-c', child]
    process = subprocess.Popen(args + ['start', str(tmp_path), durability, 'first'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    marker = tmp_path / 'boundary.json'
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and process.poll() is None:
            assert time.monotonic() < deadline, 'durable dispatch boundary was never reached'
            time.sleep(.025)
        assert marker.exists()
        # Marker file can become visible just before its final bytes are written.
        deadline = time.monotonic() + 5
        while True:
            try:
                planned = json.loads(marker.read_text())
                break
            except json.JSONDecodeError:
                assert time.monotonic() < deadline
                time.sleep(.025)
        assert count_effects(tmp_path / 'provider.db') == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
    resumed = subprocess.run(args + ['resume', str(tmp_path), durability, 'first'],
                             capture_output=True, text=True, check=False, timeout=20)
    assert resumed.returncode == 0, resumed.stderr
    outcome = json.loads(resumed.stdout.strip().splitlines()[-1])
    if boundary == 'NOT_STARTED':
        assert outcome == {'result': {'receipt': 'paid'}}
        assert count_effects(tmp_path / 'provider.db') == 1
        with sqlite3.connect(tmp_path / 'provider.db') as connection:
            dispatched_id = connection.execute('SELECT dispatch_id FROM effects').fetchone()[0]
        assert dispatched_id == planned['id']
    else:
        assert count_effects(tmp_path / 'provider.db') == 0
        assert outcome.get('error') == 'GraphExecutionUncertainError', outcome
        assert outcome['pending_tasks'][0]['id'] == planned['id']
        assert outcome['pending_tasks'][0]['status'] in ('DISPATCH_RECORDED', 'UNKNOWN')
