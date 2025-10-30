import json
import os
import uuid
from typing import List, Tuple, Dict, Any, Optional


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


def list_conversation_ids(base_dir: str = "conversations") -> List[str]:
    """Return available conversation IDs based on folders under base_dir.

    Sorted by most-recent modification time (descending).
    """
    if not os.path.isdir(base_dir):
        return []

    entries = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = 0.0
            entries.append((name, mtime))

    entries.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in entries]


def get_conversation_path(conversation_id: str, base_dir: str = "conversations") -> str:
    """Compute the path for a conversation ID (no creation)."""
    return os.path.join(base_dir, conversation_id)


def load_messages(conversation_path: str) -> List[Dict[str, Any]]:
    """Load persisted messages from messages.json if present."""
    path = os.path.join(conversation_path, "messages.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_messages(conversation_path: str, messages: List[Dict[str, Any]]) -> None:
    """Persist messages to messages.json under the conversation folder."""
    try:
        os.makedirs(conversation_path, exist_ok=True)
        path = os.path.join(conversation_path, "messages.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        # Silently ignore persistence errors to avoid breaking the chat flow
        pass


def load_papers_metadata(conversation_path: str) -> List[Dict[str, Any]]:
    """Load papers metadata if previously saved during ingestion."""
    path = os.path.join(conversation_path, "papers_metadata.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_papers_metadata(conversation_path: str, papers_metadata: Optional[List[Dict[str, Any]]]) -> None:
    """Persist papers metadata to papers_metadata.json for later reloads."""
    if not papers_metadata:
        return
    try:
        os.makedirs(conversation_path, exist_ok=True)
        path = os.path.join(conversation_path, "papers_metadata.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, ensure_ascii=False, indent=2)
    except Exception:
        # Non-fatal if we fail to persist
        pass
