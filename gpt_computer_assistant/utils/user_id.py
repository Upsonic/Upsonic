import uuid
import os


try:
    from .folder import currently_dir, artifacts_dir, media_dir

except:
    from folder import currently_dir, artifacts_dir, media_dir


user_id_db = os.path.join(artifacts_dir, "user_id.db")


def save_user_id():
    """Save a unique user ID to a file."""
    with open(user_id_db, "w") as f:
        uuid4 = str(uuid.uuid4())
        f.write(uuid4)
        return uuid4


def load_user_id():
    """Load the unique user ID from a file."""
    if not os.path.exists(user_id_db):
        return save_user_id()
    with open(user_id_db, "r") as f:
        return f.read()
