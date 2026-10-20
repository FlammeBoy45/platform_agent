import json
from pathlib import Path
from typing import Any

import pytest

from src.application.orchestrators.agent_runtime import AgentRuntime
from src.domain.entities.permissions import Role
from src.domain.protocols.logger import LoggerProtocol
from src.infrastructure.memory.json_memory_store import JsonMemoryStore
from src.infrastructure.tools.echo_tool import EchoTool
from src.infrastructure.tools.file_crud_tool import FileCrudTool
from src.infrastructure.tools.python_code_tool import PythonCodeTool
from src.infrastructure.tools.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def chat(self, messages: Any) -> str:
        if (
            messages
            and messages[0]["content"].lstrip().startswith("Tu es un planificateur.")
            and not (self.responses and self._is_plan_response(self.responses[0]))
        ):
            goal = (
                messages[0]["content"]
                .split("OBJECTIF:", 1)[-1]
                .split("\n", 1)[0]
                .strip()
            )
            encoded_goal = json.dumps(goal, ensure_ascii=False)
            return (
                f'{{"goal":{encoded_goal},"tasks":['
                f'{{"description":{encoded_goal},"depends_on":[],"status":"pending"}}]}}'
            )
        return (
            self.responses.pop(0)
            if self.responses
            else '{"tool":"final","args":{"content":"..."}}'
        )

    @staticmethod
    def _is_plan_response(response: str) -> bool:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict) and "goal" in parsed and "tasks" in parsed


class NullLogger(LoggerProtocol):
    def debug(self, m, *a):
        pass

    def info(self, m, *a):
        pass

    def warning(self, m, *a):
        pass

    def error(self, m, *a):
        pass

    def critical(self, m, *a):
        pass


async def approve_tool_action(_tool_name: str, _args: dict[str, Any]) -> bool:
    return True


@pytest.mark.asyncio
async def test_resilience_to_messy_json() -> None:
    """Vérifie que l'agent extrait le JSON même s'il est entouré de texte inutile."""
    registry = ToolRegistry()
    registry.register(EchoTool())

    messy_response = (
        "Je vais réfléchir... "
        "D'après mes calculs : "
        "```json\n"
        '{"tool": "echo", "args": {"text": "Bonjour robuste"}}\n'
        "```"
        "\nJ'espère que c'est ce que vous vouliez."
    )

    runtime = AgentRuntime(FakeLLM([messy_response]), NullLogger(), registry)
    response = await runtime.run("Repete: Bonjour robuste")

    assert response == "Bonjour robuste"


@pytest.mark.asyncio
async def test_memory_and_tool_integration(tmp_path: Path) -> None:
    """Vérifie le cycle complet : Apprentissage -> Action Tool -> Rappel de mémoire."""
    memory = JsonMemoryStore(tmp_path / "mem.json")
    registry = ToolRegistry()
    registry.register(FileCrudTool(root=tmp_path))

    # 1. On apprend le nom
    # 2. Le LLM demande une création de fichier
    # 3. Le LLM répond à la fin
    llm = FakeLLM(
        [
            '{"tool":"final","args":{"content":"C\'est noté Ada."}}',
            '{"tool":"file_crud","args":{"action":"create","path":"note.txt","content":"Fait"}}',
            '{"tool":"final","args":{"content":"Fichier créé Ada."}}',
        ]
    )

    runtime = AgentRuntime(
        llm,
        NullLogger(),
        registry,
        memory=memory,
        confirmation_handler=approve_tool_action,
    )

    # Étape 1 : Mémorisation (gérée par le runtime via regex)
    await runtime.run("Je m'appelle Ada")
    assert memory.get("user.name").value == "Ada"

    # Étape 2 : Action avec tool
    await runtime.run("Crée le fichier note.txt")
    assert (tmp_path / "note.txt").exists()

    # Étape 3 : Rappel de mémoire (court-circuite le LLM)
    response = await runtime.run("Qui suis-je ?")
    assert "Ada" in response


@pytest.mark.asyncio
async def test_security_redirection_on_forbidden_tool_access() -> None:
    """Vérifie le rejet des outils non autorisés par le contexte utilisateur."""
    registry = ToolRegistry()
    echo = EchoTool()
    echo.trigger_words = ("repete",)  # On restreint l'usage
    registry.register(echo)

    # Le LLM essaie de faire un echo alors que l'utilisateur dit juste "Salut"
    # Le runtime doit rejeter echo, car "Salut" ne contient pas "repete".
    llm = FakeLLM(
        [
            '{"tool":"echo","args":{"text":"Salut"}}',
            (
                '{"tool":"final","args":'
                '{"content":"Bonjour, comment puis-je vous aider ?"}}'
            ),
        ]
    )

    runtime = AgentRuntime(llm, NullLogger(), registry)
    response = await runtime.run("Salut !")

    # La réponse ne doit pas être "Salut" (echo) mais la réponse de correction
    assert response == "Bonjour, comment puis-je vous aider ?"


@pytest.mark.asyncio
async def test_python_tool_security_penetration() -> None:
    """Vérifie que le PythonCodeTool bloque les tentatives d'évasion courantes."""
    registry = ToolRegistry()
    registry.register(PythonCodeTool())

    # Scénario 1 : Tentative d'importation de modules système
    # Comme __import__ n'est pas dans les builtins, 'import' échouera.
    llm_attack_1 = FakeLLM(
        ['{"tool":"python_code","args":{"code":"import os; print(os.name)"}}']
    )
    runtime = AgentRuntime(
        llm_attack_1,
        NullLogger(),
        registry,
        current_role=Role.ADMIN,
    )
    response = await runtime.run("execute python: import os")

    assert response == "Import interdit: os"

    # Scénario 2 : Tentative d'accès aux fichiers via open()
    llm_attack_2 = FakeLLM(
        [
            '{"tool":"python_code","args":{"code":"f = open('
            "'src/application/orchestrators/agent_runtime.py'); "
            'print(f.read())"}}'
        ]
    )
    runtime = AgentRuntime(
        llm_attack_2,
        NullLogger(),
        registry,
        current_role=Role.ADMIN,
    )
    response = await runtime.run("python code: lis le code source")

    assert "accès aux attributs interdit" in response

    # Scénario 3 : Code légitime (doit passer)
    llm_safe = FakeLLM(
        ['{"tool":"python_code","args":{"code":"print(sum([1, 2, 3]))"}}']
    )
    runtime = AgentRuntime(
        llm_safe,
        NullLogger(),
        registry,
        current_role=Role.ADMIN,
    )
    response = await runtime.run("execute python: fais une somme")
    assert "6" in response
