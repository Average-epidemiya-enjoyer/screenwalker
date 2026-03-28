"""Unit tests for ClipboardManager.

All PyAutoGUI and pyperclip calls are mocked so tests run without a display.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from screenwalker.actions.clipboard import ClipboardManager


# ---------------------------------------------------------------------------
# ClipboardManager.read
# ---------------------------------------------------------------------------


class TestRead:
    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_returns_clipboard_text(self, mock_pyperclip: MagicMock) -> None:
        mock_pyperclip.paste.return_value = "hello world"
        cm = ClipboardManager()
        assert cm.read() == "hello world"
        mock_pyperclip.paste.assert_called_once()

    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_returns_empty_string_when_none(self, mock_pyperclip: MagicMock) -> None:
        mock_pyperclip.paste.return_value = None
        cm = ClipboardManager()
        assert cm.read() == ""

    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_returns_empty_string_when_empty(self, mock_pyperclip: MagicMock) -> None:
        mock_pyperclip.paste.return_value = ""
        cm = ClipboardManager()
        assert cm.read() == ""


# ---------------------------------------------------------------------------
# ClipboardManager.write
# ---------------------------------------------------------------------------


class TestWrite:
    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_calls_pyperclip_copy(self, mock_pyperclip: MagicMock) -> None:
        cm = ClipboardManager()
        cm.write("test text")
        mock_pyperclip.copy.assert_called_once_with("test text")

    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_write_empty_string(self, mock_pyperclip: MagicMock) -> None:
        cm = ClipboardManager()
        cm.write("")
        mock_pyperclip.copy.assert_called_once_with("")

    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_write_unicode(self, mock_pyperclip: MagicMock) -> None:
        cm = ClipboardManager()
        cm.write("Привет мир")
        mock_pyperclip.copy.assert_called_once_with("Привет мир")


# ---------------------------------------------------------------------------
# ClipboardManager.copy_selected
# ---------------------------------------------------------------------------


class TestCopySelected:
    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Windows")
    def test_without_select_all_sends_ctrl_c(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "selected"
        cm = ClipboardManager(settle_delay=0.2)
        result = cm.copy_selected(select_all=False)
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "c")
        mock_time.sleep.assert_called_once_with(0.2)
        assert result == "selected"

    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Windows")
    def test_with_select_all_sends_ctrl_a_then_ctrl_c(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "all text"
        cm = ClipboardManager()
        result = cm.copy_selected(select_all=True)
        hotkey_calls = mock_pyautogui.hotkey.call_args_list
        assert call("ctrl", "a") in hotkey_calls
        assert call("ctrl", "c") in hotkey_calls
        assert hotkey_calls.index(call("ctrl", "a")) < hotkey_calls.index(
            call("ctrl", "c")
        )
        assert result == "all text"

    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Darwin")
    def test_macos_uses_command_c(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "mac text"
        cm = ClipboardManager()
        cm.copy_selected()
        mock_pyautogui.hotkey.assert_called_once_with("command", "c")

    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Darwin")
    def test_macos_select_all_uses_command_a(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "mac all"
        cm = ClipboardManager()
        cm.copy_selected(select_all=True)
        hotkey_calls = mock_pyautogui.hotkey.call_args_list
        assert call("command", "a") in hotkey_calls
        assert call("command", "c") in hotkey_calls

    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_settle_delay_is_respected(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = ""
        cm = ClipboardManager(settle_delay=0.35)
        cm.copy_selected()
        mock_time.sleep.assert_called_once_with(0.35)


# ---------------------------------------------------------------------------
# ClipboardManager.copy_and_read
# ---------------------------------------------------------------------------


class TestCopyAndRead:
    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Windows")
    def test_clicks_then_copies(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "field text"
        cm = ClipboardManager()
        result = cm.copy_and_read(100, 200)
        mock_pyautogui.click.assert_called_once_with(100, 200)
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "c")
        assert result == "field text"

    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Windows")
    def test_copy_and_read_with_select_all(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "all field"
        cm = ClipboardManager()
        result = cm.copy_and_read(50, 75, select_all=True)
        mock_pyautogui.click.assert_called_once_with(50, 75)
        hotkey_calls = mock_pyautogui.hotkey.call_args_list
        assert call("ctrl", "a") in hotkey_calls
        assert call("ctrl", "c") in hotkey_calls
        assert result == "all field"

    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_click_precedes_copy(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Click must happen before the copy shortcut."""
        call_order: list[str] = []
        mock_pyautogui.click.side_effect = lambda *a, **kw: call_order.append("click")
        mock_pyautogui.hotkey.side_effect = lambda *a, **kw: call_order.append(
            "hotkey"
        )
        mock_pyperclip.paste.return_value = ""
        cm = ClipboardManager()
        cm.copy_and_read(10, 20)
        assert call_order[0] == "click"
        assert "hotkey" in call_order


# ---------------------------------------------------------------------------
# ClipboardManager.paste / clear
# ---------------------------------------------------------------------------


class TestPasteAndClear:
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Windows")
    def test_paste_windows(self, mock_pyautogui: MagicMock) -> None:
        ClipboardManager().paste()
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard._PLATFORM", "Darwin")
    def test_paste_macos(self, mock_pyautogui: MagicMock) -> None:
        ClipboardManager().paste()
        mock_pyautogui.hotkey.assert_called_once_with("command", "v")

    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_clear_writes_empty(self, mock_pyperclip: MagicMock) -> None:
        ClipboardManager().clear()
        mock_pyperclip.copy.assert_called_once_with("")


# ---------------------------------------------------------------------------
# ClipboardManager backward-compat alias
# ---------------------------------------------------------------------------


class TestReadSelectionAlias:
    @patch("screenwalker.actions.clipboard.time")
    @patch("screenwalker.actions.clipboard.pyautogui")
    @patch("screenwalker.actions.clipboard.pyperclip")
    def test_read_selection_delegates_to_copy_selected(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        mock_pyperclip.paste.return_value = "compat text"
        cm = ClipboardManager()
        result = cm.read_selection()
        assert result == "compat text"
