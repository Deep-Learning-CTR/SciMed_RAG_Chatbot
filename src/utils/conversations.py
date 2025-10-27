import os
import uuid
from typing import Tuple


def generate_conversation_id() -> str:
    """Generate a new unique conversation ID."""
    return uuid.uuid4().hex


def ensure_conversation_folder(base_dir: str, conversation_id: str) -> str:
    """
    Ensure a folder exists for the given conversation ID under base_dir.
    Returns the absolute path to the conversation folder.
    """
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, conversation_id)
    os.makedirs(path, exist_ok=True)
    return path


def create_new_conversation(base_dir: str = "conversations") -> Tuple[str, str]:
    """Create a new conversation folder and return (conversation_id, folder_path)."""
    conv_id = generate_conversation_id()
    path = ensure_conversation_folder(base_dir, conv_id)
    return conv_id, path

