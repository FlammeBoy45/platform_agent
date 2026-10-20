import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.application.orchestrators.agent_runtime import AgentRuntime
from src.application.orchestrators.conversation_manager import ConversationManager
from src.application.orchestrators.memory_manager import MemoryManager
from src.domain.entities.memory import MemoryFact
from src.domain.protocols.logger import LoggerProtocol
from src.infrastructure.memory import json_memory_store
from src.infrastructure.memory.json_memory_store import JsonMemoryStore
from src.infrastructure.tools.echo_tool import EchoTool
from src.infrastructure.tools.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        return prompt

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        if not self.responses:
            return '{"tool":"final","args":{"content":"ok"}}'
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


def build_runtime(memory: JsonMemoryStore) -> AgentRuntime:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return AgentRuntime(FakeLLM([]), NullLogger(), registry, memory=memory)


def test_json_memory_store_persists_facts(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = JsonMemoryStore(path)

    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")

    reloaded = JsonMemoryStore(path)
    fact = reloaded.get("user.name")

    assert fact is not None
    assert fact.value == "Ada"
    assert fact.text == "Le nom de l'utilisateur est Ada."


def test_corrupt_memory_fails_without_overwriting_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    original = '{"incomplete":'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="fichier a été conservé intact"):
        JsonMemoryStore(path)

    assert path.read_text(encoding="utf-8") == original


def test_conversation_history_stays_within_configured_limit() -> None:
    conversation = ConversationManager(max_messages=4)

    for index in range(10):
        conversation.add_user_message(f"Question {index}")
        conversation.add_assistant_message(f"Réponse {index}")

    messages = conversation.get_messages_as_dicts()
    assert len(messages) == 4
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_memory_can_forget_one_fact_or_everything(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")
    memory.remember("user.city", "Paris", "L'utilisateur vit à Paris.")

    assert memory.forget("Ada") == 1
    assert memory.get("user.name") is None
    assert memory.get("user.city") is not None
    assert memory.forget() == 1
    assert memory.all() == []


def test_memory_expires_episodes_and_facts_with_separate_retention(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    now = datetime.now(UTC)
    records = [
        MemoryFact(
            key="episode.old",
            value="old",
            text="Ancienne conversation.",
            created_at=now - timedelta(days=31),
        ),
        MemoryFact(
            key="episode.current",
            value="current",
            text="Conversation récente.",
            created_at=now - timedelta(days=29),
        ),
        MemoryFact(
            key="user.preference.old",
            value="old",
            text="Ancienne préférence.",
            created_at=now - timedelta(days=366),
        ),
        MemoryFact(
            key="user.preference.current",
            value="current",
            text="Préférence récente.",
            created_at=now - timedelta(days=364),
        ),
    ]
    path.write_text(
        json.dumps(
            [record.model_dump(mode="json") for record in records],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory = JsonMemoryStore(path)

    assert [fact.key for fact in memory.all()] == [
        "episode.current",
        "user.preference.current",
    ]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert {fact["key"] for fact in persisted} == {
        "episode.current",
        "user.preference.current",
    }


def test_legacy_memory_entries_receive_a_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            [
                {
                    "key": "user.name",
                    "value": "Ada",
                    "text": "Le nom de l'utilisateur est Ada.",
                }
            ]
        ),
        encoding="utf-8",
    )

    fact = JsonMemoryStore(path).get("user.name")
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert fact is not None
    assert fact.created_at.tzinfo is not None
    assert fact.created_at >= datetime.now(UTC) - timedelta(seconds=5)
    assert "created_at" in persisted[0]


def test_memory_retention_limit_keeps_personal_facts(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json", max_entries=3)
    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")
    for index in range(5):
        memory.store_episode(f"Question {index}", f"Réponse {index}")

    assert len(memory.all()) == 3
    assert memory.get("user.name") is not None
    assert len([fact for fact in memory.all() if fact.key.startswith("episode.")]) == 2


def test_json_memory_store_retries_windows_replace_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "memory.json"
    memory = JsonMemoryStore(path)
    replace = json_memory_store.os.replace
    attempts = 0
    temporary_names: list[str] = []

    def fail_once(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        temporary_names.append(Path(source).name)
        if attempts == 1:
            raise PermissionError(5, "Access is denied")
        replace(source, destination)

    monkeypatch.setattr(json_memory_store.os, "replace", fail_once)

    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")

    assert attempts == 2
    assert temporary_names[0] != "memory.json.tmp"
    assert len(set(temporary_names)) == 1
    assert JsonMemoryStore(path).get("user.name").value == "Ada"


def test_memory_manager_persists_explicit_memory_and_preferences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    memory = JsonMemoryStore(path)
    manager = MemoryManager(memory)

    manager.learn_facts("Retiens que je préfère les réponses courtes.")
    manager.learn_facts("Je préfère le thé vert.")

    reloaded = JsonMemoryStore(path)
    facts = reloaded.all()

    assert any("réponses courtes" in fact.text for fact in facts)
    assert any("thé vert" in fact.text for fact in facts)
    assert reloaded.search("mes préférences thé vert")


def test_json_memory_store_persists_conversation_episodes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    memory = JsonMemoryStore(path)
    memory.store_episode(
        "Je travaille sur un agent Python.",
        {"type": "final", "output": "Je retiens le projet."},
    )

    reloaded = JsonMemoryStore(path)
    recent = reloaded.list_recent()

    assert len(recent) == 1
    assert "agent Python" in recent[0].text
    assert "Je retiens le projet." in recent[0].text
    assert reloaded.retrieve("agent Python")


@pytest.mark.asyncio
async def test_runtime_remembers_user_name_after_short_term_reset(
    tmp_path: Path,
) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    runtime = build_runtime(memory)

    await runtime.run("mon nom est Ada")
    runtime.reset()
    response = await runtime.run("quel est mon nom ?")

    assert response == "Ton nom est Ada."


@pytest.mark.asyncio
async def test_runtime_can_forget_a_saved_name(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    runtime = build_runtime(memory)

    await runtime.run("mon nom est Ada")
    response = await runtime.run("oublie mon nom")

    assert "oublié" in response
    assert memory.get("user.name") is None


@pytest.mark.asyncio
async def test_runtime_can_forget_all_persistent_memory(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")
    memory.store_episode("question", "réponse")
    runtime = build_runtime(memory)

    response = await runtime.run("oublie tout")

    assert "oublié" in response
    assert memory.all() == []


def test_runtime_passes_persistent_memory_to_planner(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")

    runtime = build_runtime(memory)

    assert runtime.planner.memory_v2 is memory


@pytest.mark.asyncio
async def test_memory_is_shared_after_recreating_runtime_with_another_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first_memory = JsonMemoryStore(path)
    first_runtime = build_runtime(first_memory)

    await first_runtime.run("je m'appelle Ada")

    reloaded_memory = JsonMemoryStore(path)
    second_runtime = build_runtime(reloaded_memory)
    response = await second_runtime.run("quel est mon nom ?")

    assert response == "Ton nom est Ada."


@pytest.mark.asyncio
async def test_user_can_display_persistent_memory(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")
    memory.store_episode(
        "Je construis un agent Python.",
        {"type": "final", "output": "On avance étape par étape."},
    )

    response = await build_runtime(memory).run("montre-moi ta mémoire")

    assert "souvenirs récents" in response
    assert "agent Python" in response
    assert "Ada" in response


@pytest.mark.asyncio
async def test_runtime_persists_each_completed_exchange(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    runtime = build_runtime(memory)

    await runtime.run("Je travaille sur un agent conversationnel.")

    reloaded = JsonMemoryStore(tmp_path / "memory.json")
    episodes = [
        fact for fact in reloaded.list_recent() if fact.key.startswith("episode.")
    ]
    assert len(episodes) == 1
    assert "agent conversationnel" in episodes[0].text
    assert "ok" in episodes[0].text


@pytest.mark.asyncio
async def test_model_response_does_not_create_persistent_facts(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    llm = FakeLLM(
        [
            '{"goal":"question","tasks":[{"description":"question",'
            '"depends_on":[],"status":"pending"}]}',
            '{"tool":"final","args":{"content":"Python est un langage."}}',
        ]
    )
    runtime = AgentRuntime(llm, NullLogger(), ToolRegistry(), memory=memory)

    await runtime.run("quel langage est Python ?")

    assert not any(fact.key.startswith("fact.") for fact in memory.all())


@pytest.mark.asyncio
async def test_explicit_memory_commands_save_current_and_previous_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    memory = JsonMemoryStore(path)
    llm = FakeLLM([])
    runtime = AgentRuntime(llm, NullLogger(), ToolRegistry(), memory=memory)

    first_response = await runtime.run(
        "souviens toi que je suis isi la flamme ton pere et toi cest ava"
    )
    second_response = await runtime.run("ecris le dans tes memoires")

    reloaded = JsonMemoryStore(path)
    saved = reloaded.search("isi la flamme ava")

    assert "enregistré" in first_response
    assert "enregistré" in second_response
    assert any("isi la flamme" in fact.text for fact in saved)
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_agent_uses_remembered_assistant_name_after_reload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    runtime = build_runtime(JsonMemoryStore(path))

    await runtime.run("souviens toi que je suis isi la flamme ton pere et toi cest ava")

    reloaded_runtime = build_runtime(JsonMemoryStore(path))
    response = await reloaded_runtime.run("qui est tu")

    assert response == "Je suis ava, ton assistant IA."


@pytest.mark.asyncio
async def test_informal_identity_statement_restores_both_names_after_reload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first_runtime = build_runtime(JsonMemoryStore(path))

    await first_runtime.run("je suis isi la flamme ton pere et toi ces ava")

    reloaded_memory = JsonMemoryStore(path)
    reloaded_runtime = build_runtime(reloaded_memory)

    assert reloaded_memory.get("user.name").value == "isi la flamme"
    assert reloaded_memory.get("agent.name").value == "ava"
    assert await reloaded_runtime.run("qui je suis") == ("Ton nom est isi la flamme.")
    assert await reloaded_runtime.run("ton nom") == ("Je suis ava, ton assistant IA.")


@pytest.mark.asyncio
async def test_identity_and_father_questions_use_persisted_relationships(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    memory = JsonMemoryStore(path)
    runtime = build_runtime(memory)

    await runtime.run("je suis isi la flamme ton pere et toi ces ava")

    assert memory.get("agent.father").value == "isi la flamme"
    assert await runtime.run("qui est ava") == ("Ava, c'est moi, ton assistant IA.")
    assert await runtime.run("et sui est son pere") == (
        "Tu m'as dit que isi la flamme est mon père."
    )
    assert await runtime.run("qui est ton pere") == (
        "Tu m'as dit que isi la flamme est mon père."
    )
    assert await runtime.run("et je suis pas la peine d'en rajouter") == (
        "Compris, je n'en rajoute pas."
    )


def test_father_of_agent_statement_accepts_common_typo(
    tmp_path: Path,
) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    manager = MemoryManager(memory)

    manager.learn_facts("je suis lle pere de ava")

    assert memory.get("agent.name").value == "ava"
    assert memory.get("agent.father").value == "toi, l'utilisateur"


def test_existing_identity_memory_is_migrated_on_load(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    memory.remember(
        "user.fact.legacy",
        "je suis isi la flamme ton pere et toi cest ava",
        "L'utilisateur a demandé de retenir : "
        "je suis isi la flamme ton pere et toi cest ava.",
    )

    MemoryManager(memory)

    assert memory.get("user.name").value == "isi la flamme"
    assert memory.get("agent.name").value == "ava"


@pytest.mark.asyncio
async def test_save_memory_request_does_not_claim_success_when_disabled() -> None:
    llm = FakeLLM([])
    runtime = AgentRuntime(
        llm,
        NullLogger(),
        ToolRegistry(),
        enable_memory=False,
    )

    response = await runtime.run("ecris le dans tes memoires")

    assert "désactivée" in response
    assert llm.call_count == 0
