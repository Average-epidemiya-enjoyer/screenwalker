"""Action primitives — mouse, keyboard, and clipboard operations.

All action functions are thin wrappers around PyAutoGUI / pyperclip that add
platform normalisation, error handling, and optional dry-run no-op mode.
"""

from screenwalker.actions.clipboard import ClipboardManager
from screenwalker.actions.keyboard import KeyboardController
from screenwalker.actions.mouse import MouseController

__all__ = ["MouseController", "KeyboardController", "ClipboardManager"]
