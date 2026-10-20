import pytest

from src.application.orchestrators.intent_classifier import (
    get_forced_tool_for_intent,
    has_action_intent,
    is_plan_resume_request,
)


@pytest.mark.parametrize(
    ("user_input", "expected_tool"),
    [
        ("crée un fichier notes.txt", "file_crud"),
        ("calcule 18 fois 24", "calculator"),
        ("quel est le prix du bitcoin", "crypto_price"),
        ("quelle heure est-il en UTC+2", "datetime"),
        ("cherche dans le code la classe AgentRuntime", "workspace_search"),
        ("cherche des informations sur les volcans", "google_search"),
        ("écris une fonction Python", "python_code"),
    ],
)
def test_forced_tool_matches_intent(user_input: str, expected_tool: str) -> None:
    tool_name, _ = get_forced_tool_for_intent(user_input)

    assert tool_name == expected_tool


@pytest.mark.parametrize(
    ("user_input", "expected"),
    [
        ("reprends la tâche", True),
        ("continue, s'il te plaît", True),
        ("bonjour", False),
    ],
)
def test_plan_resume_request(user_input: str, expected: bool) -> None:
    assert is_plan_resume_request(user_input) is expected


@pytest.mark.parametrize(
    ("user_input", "expected"),
    [
        ("crée un fichier", True),
        ("calcule 2 + 2", True),
        ("bonjour", False),
    ],
)
def test_action_intent(user_input: str, expected: bool) -> None:
    assert has_action_intent(user_input) is expected
