"""
memory.py
=========
Session memory backed only by langchain_core — no langchain_community required.
"""
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage


class SessionMemory:
    """
    Lightweight in-process message store.

    Keeps a list of ``BaseMessage`` objects that can be passed directly to
    ``llm.invoke(messages)``.  Used internally by ``FinancialAdvisorSession``
    but exposed here for testing or direct use.

    Args:
        system_prompt (str): Optional system message prepended to every
                             conversation.  Pass ``""`` to omit.
    """

    def __init__(self, system_prompt: str = "") -> None:
        self._messages: List[BaseMessage] = (
            [SystemMessage(content=system_prompt)] if system_prompt else []
        )

    def add(self, message: BaseMessage) -> None:
        """Append a single message to the store."""
        self._messages.append(message)

    def all(self) -> List[BaseMessage]:
        """Return a copy of the full message list."""
        return list(self._messages)

    def clear(self, system_prompt: str = "") -> None:
        """Reset to an empty store (or just the system prompt if provided)."""
        self._messages = (
            [SystemMessage(content=system_prompt)] if system_prompt else []
        )

    def __len__(self) -> int:
        return len(self._messages)


def get_memory(system_prompt: str = "") -> SessionMemory:
    """
    Create and return a fresh ``SessionMemory`` instance.

    Args:
        system_prompt (str): Optional system message to seed the memory.

    Returns:
        SessionMemory
    """
    return SessionMemory(system_prompt=system_prompt)