import os
import sqlite3
import json
from dotenv import load_dotenv
import signal
from .folder import BASE_PATH


class ConfigManager:
    def __init__(self, db_name="config.sqlite"):
        self.db_path = os.path.join(BASE_PATH, db_name)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS config_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        self.conn.commit()
        
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        if hasattr(self, 'conn'):
            try:
                self.conn.commit()
                self.conn.close()
            except:
                pass
        signal.default_int_handler(signum, frame)

    def initialize(self, key):
        load_dotenv()
        value = os.getenv(key)
        if value is not None:
            self.set(key, value)

    def get(self, key, default=None):
        try:
            self.cursor.execute('SELECT value FROM config_store WHERE key = ?', (key,))
            result = self.cursor.fetchone()
            return json.loads(result[0]) if result else default
        except:
            return default

    def delete(self, key):
        try:
            self.cursor.execute('DELETE FROM config_store WHERE key = ?', (key,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except:
            return False

    def set(self, key, value):
        try:
            value_json = json.dumps(value)
            self.cursor.execute('REPLACE INTO config_store (key, value) VALUES (?, ?)',
                              (key, value_json))
            self.conn.commit()
            return True
        except:
            return False

    def dump(self):
        try:
            self.conn.commit()
            return True
        except:
            return False

    def __del__(self):
        if hasattr(self, 'conn'):
            try:
                self.conn.commit()
                self.conn.close()
            except:
                pass


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

ClientConfiguration = ConfigManager(db_name="client_config.sqlite")