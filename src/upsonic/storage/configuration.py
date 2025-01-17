import os
from dotenv import load_dotenv
import pickledb
from .folder import BASE_PATH


class ConfigManager:
    def __init__(self, db_name="config.db"):
        db_path = os.path.join(BASE_PATH, db_name)
        self.db = pickledb.load(db_path, False)

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

ClientConfiguration = ConfigManager(db_name="client_config.db")