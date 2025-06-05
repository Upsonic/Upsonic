import uuid
from .storage.configuration import Configuration


def generate_system_id() -> str:
    """
    Generate a unique system ID and store it.

    Returns:
        The generated unique system ID as a string.
    """
    system_id = str(uuid.uuid4())
    return system_id 


def get_system_id() -> str:
    """
    Get the system ID from the configuration.
    """
    the_system_id = Configuration.get("system_id")
    if the_system_id is None:
        the_system_id = generate_system_id()
        Configuration.set("system_id", the_system_id)
    return the_system_id