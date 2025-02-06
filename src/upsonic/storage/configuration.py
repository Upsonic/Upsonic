import os
from dotenv import load_dotenv
import pickledb
import signal
from .folder import BASE_PATH


class ConfigManager:
    def __init__(self, db_name="config.db"):
        db_path = os.path.join(BASE_PATH, db_name)
        self.db = pickledb.load(db_path, False)
        # Override pickledb's signal handler with our own
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle termination signals by dumping the database before exit"""
        if hasattr(self, 'db'):
            self.db.dump()
        # Re-raise the signal
        signal.default_int_handler(signum, frame)

    def initialize(self, key):
        load_dotenv()
        value = os.getenv(key)
        if value is not None:
            self.set(key, value)

    def get(self, key, default=None):
        value = self.db.get(key)
        return value if value is not False else default

    def set(self, key, value):
        self.db.set(key, value)
        self.db.dump()


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

ClientConfiguration = ConfigManager(db_name="client_config.db")