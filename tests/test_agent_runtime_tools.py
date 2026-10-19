from pathlib import Path
from typing import Any

import pytest

import src.application.orchestrators.agent_runtime as agent_runtime_module
from src.application.orchestrators.agent_runtime import AgentRuntime
from src.config.settings import Settings
from src.domain.protocols.logger import LoggerProtocol
from src.infrastructure.tools.calculator_tool import CalculatorTool
from src.infrastructure.tools.datetime_tool import DateTimeTool
from src.infrastructure.tools.default_tools import register_default_tools
from src.infrastructure.tools.echo_tool import EchoTool
from src.infrastructure.tools.file_crud_tool import FileCrudTool
from src.infrastructure.tools.file_list_tool import FileListTool
from src.infrastructure.tools.text_stats_tool import TextStatsTool
from src.infrastructure.tools.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, str]]] = []

    async def generate(self, prompt: str) -> str:
        return prompt

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        if not self.responses:
            return '{"tool":"final","args":{"content":"fallback"}}'
        return self.responses.pop(0)

    async def stream(self, messages: list[dict[str, str]]) -> Any:
        return None

    async def close(self) -> None:
        return None


class NullLogger(LoggerProtocol):
    def debug(self, message: str, *args: object) -> None:
        pass

    def info(self, message: str, *args: object) -> None:
        pass

    def warning(self, message: str, *args: object) -> None:
        pass

    def error(self, message: str, *args: object) -> None:
        pass

    def critical(self, message: str, *args: object) -> None:
        pass


def build_runtime(responses: list[str]) -> AgentRuntime:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return AgentRuntime(FakeLLM(responses), NullLogger(), registry)


def build_runtime_with_utilities(responses: list[str]) -> AgentRuntime:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(TextStatsTool())
    return AgentRuntime(FakeLLM(responses), NullLogger(), registry)


@pytest.mark.asyncio
async def test_extracts_json_when_model_wraps_it_in_text() -> None:
    runtime = build_runtime(
        [
            'Voici la reponse: {"tool":"final","args":{"content":"Bonjour"}}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Bonjour"


@pytest.mark.asyncio
async def test_accepts_response_text_final_variant() -> None:
    runtime = build_runtime(
        [
            '{"response":"final","text":"Bonjour !"}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Bonjour !"


@pytest.mark.asyncio
async def test_accepts_message_final_variant() -> None:
    runtime = build_runtime(
        [
            '{"message":"Salut ISI la flamme !"}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Salut ISI la flamme !"


@pytest.mark.asyncio
async def test_echo_returns_directly_when_requested() -> None:
    runtime = build_runtime(
        [
            '{"tool":"echo","args":{"text":"bonjour agent"}}',
        ]
    )

    response = await runtime.run("repete exactement: bonjour agent")

    assert response == "bonjour agent"


@pytest.mark.asyncio
async def test_echo_args_are_filled_from_user_input_when_missing() -> None:
    runtime = build_runtime(
        [
            '{"tool":"echo","args":{}}',
        ]
    )

    response = await runtime.run("echo: test 123")

    assert response == "test 123"


@pytest.mark.asyncio
async def test_unknown_tool_is_converted_to_final_response() -> None:
    runtime = build_runtime(
        [
            '{"tool":"helloworld","args":{}}',
        ]
    )

    response = await runtime.run("utilise le tool helloworld")

    assert response == (
        'Je n\'ai pas acces au tool "helloworld". Tools disponibles: echo.'
    )


@pytest.mark.asyncio
async def test_echo_is_rejected_for_plain_conversation() -> None:
    runtime = build_runtime(
        [
            '{"tool":"echo","args":{"text":"salut"}}',
            '{"tool":"final","args":{"content":"Salut !"}}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Salut !"


@pytest.mark.asyncio
async def test_low_quality_final_marker_is_retried() -> None:
    runtime = build_runtime(
        [
            '{"tool":"final","args":{"content":"final"}}',
            '{"tool":"final","args":{"content":"Salut, a bientot !"}}',
        ]
    )

    response = await runtime.run("salut et bye")

    assert response == "Salut, a bientot !"


@pytest.mark.asyncio
async def test_none_tool_is_retried_as_final_response() -> None:
    runtime = build_runtime(
        [
            '{"tool":"None","args":{}}',
            '{"tool":"final","args":{"content":"Salut !"}}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Salut !"


@pytest.mark.asyncio
async def test_none_final_content_is_retried() -> None:
    runtime = build_runtime(
        [
            '{"tool":"final","args":{"content":"None"}}',
            '{"tool":"final","args":{"content":"Salut !"}}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Salut !"


@pytest.mark.asyncio
async def test_plain_plan_final_text_is_retried() -> None:
    runtime = build_runtime(
        [
            "Plan actuel :\nPas encore de tâche à effectuer.\n\nFinal",
            '{"tool":"final","args":{"content":"Salut !"}}',
        ]
    )

    response = await runtime.run("salut")

    assert response == "Salut !"


@pytest.mark.asyncio
async def test_bare_low_quality_words_are_retried() -> None:
    runtime = build_runtime(
        [
            "plan",
            '{"tool":"final","args":{"content":"Je reste simple."}}',
        ]
    )

    response = await runtime.run("de fois tu dis plan pour rien")

    assert response == "Je reste simple."


@pytest.mark.asyncio
async def test_plan_dict_without_tool_is_retried() -> None:
    runtime = build_runtime(
        [
            (
                '{"goal":"Lister les fichiers",'
                '"tasks":[{"description":"x","status":"pending"}]}'
            ),
            (
                '{"tool":"final",'
                '"args":{"content":"Je peux lister les tools disponibles."}}'
            ),
        ]
    )

    response = await runtime.run("liste des fichiers")

    assert response == "Je peux lister les tools disponibles."


@pytest.mark.asyncio
async def test_tools_list_is_answered_without_llm_tool_confusion() -> None:
    runtime = build_runtime([])

    response = await runtime.run("liste des tools disponibles")

    assert response.startswith("Tools disponibles :")
    assert "- echo:" in response


def test_default_tools_include_web_fetch() -> None:
    registry = ToolRegistry()

    settings = Settings(
        WORKSPACE_PATH="sandbox",
        ENABLE_PYTHON_CODE=False,
        SERPAPI_API_KEY="",
    )
    register_default_tools(registry, settings)

    assert registry.get("web_fetch") is not None
    assert registry.get("google_search") is not None
    assert registry.get("python_code") is None
    file_crud = registry.get("file_crud")
    file_list = registry.get("file_list")
    assert isinstance(file_crud, FileCrudTool)
    assert isinstance(file_list, FileListTool)
    assert file_crud.root.name == "sandbox"
    assert file_list.root.name == "sandbox"


def test_python_code_tool_is_opt_in() -> None:
    registry = ToolRegistry()
    settings = Settings(ENABLE_PYTHON_CODE=True, SERPAPI_API_KEY="")

    register_default_tools(registry, settings)

    assert registry.get("python_code") is not None


@pytest.mark.asyncio
async def test_memory_can_be_disabled_without_initializing_persistent_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_initialized() -> None:
        raise AssertionError("MemoryV2 should not initialize when memory is disabled")

    monkeypatch.setattr(agent_runtime_module, "MemoryV2", fail_if_initialized)
    runtime = AgentRuntime(
        FakeLLM([
            '{"goal":"salut","tasks":[{"description":"salut","status":"pending"}]}',
            '{"tool":"final","args":{"content":"Bonjour !"}}',
        ]),
        NullLogger(),
        ToolRegistry(),
        enable_memory=False,
    )

    response = await runtime.run("salut")

    assert response == "Bonjour !"


@pytest.mark.asyncio
async def test_informal_module_request_creates_empty_python_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoMemory:
        def __init__(self) -> None:
            pass

        def store_episode(self, *args: Any, **kwargs: Any) -> None:
            pass

        def store_fact(self, *args: Any, **kwargs: Any) -> None:
            pass

        def store_skill(self, *args: Any, **kwargs: Any) -> None:
            pass

        def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        def retrieve_skills(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    monkeypatch.setattr(agent_runtime_module, "MemoryV2", NoMemory)
    registry = ToolRegistry()
    registry.register(FileCrudTool(root=tmp_path))
    runtime = AgentRuntime(
        FakeLLM([
            (
                '{"goal":"module auth",'
                '"tasks":[{"description":"module auth","status":"pending"}]}'
            ),
            '{"tool":"final","args":{"content":"Je vais preparer le module."}}',
        ]),
        NullLogger(),
        registry,
        enable_memory=True,
    )

    result = await runtime.run("fais-moi le modou fichier auth meme vide stp")

    assert result == "Fichier cree: auth.py"
    assert (tmp_path / "auth.py").read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_calculator_returns_directly() -> None:
    runtime = build_runtime_with_utilities(
        [
            '{"tool":"calculator","args":{"expression":"2 + 2 * 3"}}',
        ]
    )

    response = await runtime.run("calcule 2 + 2 * 3")

    assert response == "8"


@pytest.mark.asyncio
async def test_calculator_still_runs_when_plan_uses_done_status() -> None:
    runtime = build_runtime_with_utilities(
        [
            (
                '{"tool":"calculator","args":{"expression":"9 ** 5"},'
                '"plan":{"goal":"Calculer 9 puissance 5",'
                '"tasks":[{"description":"Evaluer 9 ** 5","status":"done"}]}}'
            ),
        ]
    )

    response = await runtime.run("calcul 9 puissance 5")

    assert response == "59049"


@pytest.mark.asyncio
async def test_datetime_accepts_europe_paris_from_model() -> None:
    runtime = build_runtime_with_utilities(
        [
            '{"tool":"datetime","args":{"timezone":"Europe/Paris"}}',
        ]
    )

    response = await runtime.run("donne lheure utc +2")

    assert "Timezone inconnue" not in response
    assert "UTC+02:00" in response or "CEST" in response or "CET" in response


@pytest.mark.asyncio
async def test_text_stats_fills_missing_text_from_user_input() -> None:
    runtime = build_runtime_with_utilities(
        [
            '{"tool":"text_stats","args":{}}',
        ]
    )

    response = await runtime.run("compte: bonjour agent")

    assert response == "Caracteres: 13; Mots: 2; Lignes: 1"
