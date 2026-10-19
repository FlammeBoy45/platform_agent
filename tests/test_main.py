import asyncio

import pytest

from src.main import _confirm_tool_action, _conversation_loop


class CancelledAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, user_input: str) -> str:
        self.calls += 1
        raise asyncio.CancelledError


@pytest.mark.parametrize("stop_signal", [KeyboardInterrupt, EOFError])
@pytest.mark.asyncio
async def test_terminal_stop_at_prompt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop_signal: type[BaseException],
) -> None:
    def fake_input(_prompt: str) -> str:
        raise stop_signal

    monkeypatch.setattr("builtins.input", fake_input)
    agent = CancelledAgent()

    await _conversation_loop(agent)

    output = capsys.readouterr().out
    assert agent.calls == 0
    assert "Traceback" not in output
    assert "Requête interrompue" not in output
    assert output.count("Agent arrêté.") == 1


@pytest.mark.asyncio
async def test_ctrl_c_during_response_exits_chat_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "question")
    agent = CancelledAgent()

    await _conversation_loop(agent)

    output = capsys.readouterr().out
    assert agent.calls == 1
    assert "Traceback" not in output
    assert "Tu peux continuer" not in output
    assert "Agent arrêté." in output


@pytest.mark.parametrize("answer", ["o", "oui", "Y", "yes"])
@pytest.mark.asyncio
async def test_confirmation_accepts_explicit_yes(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    assert await _confirm_tool_action(
        "file_crud",
        {"action": "delete", "path": "notes.txt"},
    )


@pytest.mark.parametrize("answer", ["", "non", "no"])
@pytest.mark.asyncio
async def test_confirmation_defaults_to_no(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    assert not await _confirm_tool_action(
        "file_crud",
        {"action": "delete", "path": "notes.txt"},
    )
