"""Unit tests for KeyboardController.

All PyAutoGUI and pyperclip calls are mocked so tests run without a display.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from screenwalker.actions.keyboard import KeyboardController, _normalise_key


# ---------------------------------------------------------------------------
# _normalise_key helpers
# ---------------------------------------------------------------------------


class TestNormaliseKey:
    def test_alias_return_maps_to_enter(self) -> None:
        assert _normalise_key("return") == "enter"

    def test_alias_esc_maps_to_escape(self) -> None:
        assert _normalise_key("esc") == "escape"

    def test_alias_del_maps_to_delete(self) -> None:
        assert _normalise_key("del") == "delete"

    def test_alias_pgup_maps_to_pageup(self) -> None:
        assert _normalise_key("pgup") == "pageup"

    def test_unknown_key_returned_lowercase(self) -> None:
        assert _normalise_key("Shift") == "shift"

    def test_already_canonical_key_unchanged(self) -> None:
        assert _normalise_key("enter") == "enter"


# ---------------------------------------------------------------------------
# KeyboardController.type_text
# ---------------------------------------------------------------------------


class TestTypeText:
    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_ascii_text_uses_typewrite(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController(typing_interval=0.05)
        kb.type_text("hello world", interval=0.05)
        mock_pyautogui.typewrite.assert_called_once_with("hello world", interval=0.05)

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_uses_instance_interval_when_none_passed(
        self, mock_pyautogui: MagicMock
    ) -> None:
        kb = KeyboardController(typing_interval=0.07)
        kb.type_text("abc")
        mock_pyautogui.typewrite.assert_called_once_with("abc", interval=0.07)

    @patch("screenwalker.actions.keyboard.pyperclip")
    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_cyrillic_uses_clipboard_paste(
        self, mock_pyautogui: MagicMock, mock_pyperclip: MagicMock
    ) -> None:
        kb = KeyboardController()
        kb.type_text("привет")
        mock_pyperclip.copy.assert_called_once_with("привет")
        mock_pyautogui.hotkey.assert_called_once()
        assert "v" in mock_pyautogui.hotkey.call_args.args

    @patch("screenwalker.actions.keyboard.pyperclip")
    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_mixed_ascii_unicode_uses_clipboard(
        self, mock_pyautogui: MagicMock, mock_pyperclip: MagicMock
    ) -> None:
        kb = KeyboardController()
        kb.type_text("hello мир")
        mock_pyperclip.copy.assert_called_once_with("hello мир")
        mock_pyautogui.typewrite.assert_not_called()

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_empty_string_calls_typewrite(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.type_text("")
        mock_pyautogui.typewrite.assert_called_once_with("", interval=kb.typing_interval)


# ---------------------------------------------------------------------------
# KeyboardController.type_unicode
# ---------------------------------------------------------------------------


class TestTypeUnicode:
    @patch("screenwalker.actions.keyboard.pyperclip")
    @patch("screenwalker.actions.keyboard.pyautogui")
    @patch("screenwalker.actions.keyboard._PLATFORM", "Windows")
    def test_windows_uses_ctrl_v(
        self, mock_pyautogui: MagicMock, mock_pyperclip: MagicMock
    ) -> None:
        kb = KeyboardController()
        kb.type_unicode("Тест")
        mock_pyperclip.copy.assert_called_once_with("Тест")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch("screenwalker.actions.keyboard.pyperclip")
    @patch("screenwalker.actions.keyboard.pyautogui")
    @patch("screenwalker.actions.keyboard._PLATFORM", "Darwin")
    def test_macos_uses_command_v(
        self, mock_pyautogui: MagicMock, mock_pyperclip: MagicMock
    ) -> None:
        kb = KeyboardController()
        kb.type_unicode("Тест")
        mock_pyautogui.hotkey.assert_called_once_with("command", "v")


# ---------------------------------------------------------------------------
# KeyboardController.hotkey
# ---------------------------------------------------------------------------


class TestHotkey:
    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_hotkey_passes_keys_to_pyautogui(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.hotkey("ctrl", "s")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "s")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_hotkey_normalises_aliases(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.hotkey("ESC")
        mock_pyautogui.hotkey.assert_called_once_with("escape")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_hotkey_three_keys(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.hotkey("ctrl", "alt", "del")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "alt", "delete")


# ---------------------------------------------------------------------------
# KeyboardController.press
# ---------------------------------------------------------------------------


class TestPress:
    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_press_canonical_key(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.press("enter")
        mock_pyautogui.press.assert_called_once_with("enter")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_press_normalises_alias(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.press("return")
        mock_pyautogui.press.assert_called_once_with("enter")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_press_uppercase_normalised(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.press("Tab")
        mock_pyautogui.press.assert_called_once_with("tab")


# ---------------------------------------------------------------------------
# KeyboardController.key_down / key_up
# ---------------------------------------------------------------------------


class TestKeyDownUp:
    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_key_down(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.key_down("shift")
        mock_pyautogui.keyDown.assert_called_once_with("shift")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_key_up(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.key_up("shift")
        mock_pyautogui.keyUp.assert_called_once_with("shift")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_key_down_normalises(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.key_down("CTRL")
        mock_pyautogui.keyDown.assert_called_once_with("ctrl")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_hold_and_release_sequence(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.key_down("shift")
        kb.key_up("shift")
        assert mock_pyautogui.keyDown.call_count == 1
        assert mock_pyautogui.keyUp.call_count == 1


# ---------------------------------------------------------------------------
# KeyboardController compound helpers
# ---------------------------------------------------------------------------


class TestCompoundHelpers:
    @patch("screenwalker.actions.keyboard.pyautogui")
    @patch("screenwalker.actions.keyboard._PLATFORM", "Windows")
    def test_select_all_windows(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.select_all()
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "a")

    @patch("screenwalker.actions.keyboard.pyautogui")
    @patch("screenwalker.actions.keyboard._PLATFORM", "Darwin")
    def test_select_all_macos(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.select_all()
        mock_pyautogui.hotkey.assert_called_once_with("command", "a")

    @patch("screenwalker.actions.keyboard.pyautogui")
    def test_clear_field_selects_then_deletes(self, mock_pyautogui: MagicMock) -> None:
        kb = KeyboardController()
        kb.clear_field()
        assert mock_pyautogui.hotkey.call_count == 1  # select_all
        mock_pyautogui.press.assert_called_once_with("delete")
