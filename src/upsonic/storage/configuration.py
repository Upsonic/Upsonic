import os
import sqlite3
import json
from dotenv import load_dotenv
import signal
import sys
import threading
import logging
from contextlib import contextmanager
from .folder import BASE_PATH


class ConfigManager:
    def __init__(self, db_name="config.sqlite"):
        self.db_path = os.path.join(BASE_PATH, db_name)
        self._local = threading.local()
        self._setup_database()
        
        # Only set up signal handlers if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGTERM, self._handle_signal)
                signal.signal(signal.SIGINT, self._handle_signal)
            except ValueError:
                # Ignore signal handling errors if we can't set them up
                pass

    def _setup_database(self):
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS config_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            conn.commit()

    @contextmanager
    def _get_connection(self):
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
        try:
            yield self._local.conn
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise

    def _handle_signal(self, signum, frame):
        self.close_all_connections()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    def close_all_connections(self):
        if hasattr(self._local, 'conn'):
            try:
                self._local.conn.commit()
                self._local.conn.close()
                del self._local.conn
            except Exception as e:
                logging.error(f"Error closing connection: {e}")

    def initialize(self, key):
        load_dotenv()
        value = os.getenv(key)
        if value is not None:
            self.set(key, value)

    def get(self, key, default=None):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM config_store WHERE key = ?', (key,))
                result = cursor.fetchone()
                return json.loads(result[0]) if result else default
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logging.error(f"Error retrieving key {key}: {e}")
            return default

    def delete(self, key):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM config_store WHERE key = ?', (key,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Error deleting key {key}: {e}")
            return False

    def set(self, key, value):
        try:
            value_json = json.dumps(value)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('REPLACE INTO config_store (key, value) VALUES (?, ?)',
                          (key, value_json))
                conn.commit()
                return True
        except (sqlite3.Error, json.JSONEncodeError) as e:
            logging.error(f"Error setting key {key}: {e}")
            return False

    def dump(self):
        try:
            with self._get_connection() as conn:
                conn.commit()
                return True
        except sqlite3.Error as e:
            logging.error(f"Error dumping database: {e}")
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all_connections()

    def __del__(self):
        self.close_all_connections()


# Create a single instance of ConfigManager
Configuration = ConfigManager()

Configuration.initialize("OPENAI_API_KEY")
Configuration.initialize("ANTHROPIC_API_KEY")
Configuration.initialize("AZURE_OPENAI_ENDPOINT")
Configuration.initialize("AZURE_OPENAI_API_VERSION")
Configuration.initialize("AZURE_OPENAI_API_KEY")
Configuration.initialize("AWS_ACCESS_KEY_ID")
Configuration.initialize("AWS_SECRET_ACCESS_KEY")
Configuration.initialize("AWS_REGION")
Configuration.initialize("DEEPSEEK_API_KEY")
Configuration.initialize("GOOGLE_GLA_API_KEY")
Configuration.initialize("OPENROUTER_API_KEY")

Configuration.initialize("OLLAMA_BASE_URL")

ClientConfiguration = ConfigManager(db_name="client_config.sqlite")