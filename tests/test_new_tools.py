from pathlib import Path

import pytest

from src.domain.entities.permissions import PermissionManager, Role
from src.infrastructure.memory.json_memory_store import JsonMemoryStore
from src.infrastructure.tools.default_tools import register_default_tools
from src.infrastructure.tools.memory_tool import MemoryTool
from src.infrastructure.tools.tool_registry import ToolRegistry
from src.infrastructure.tools.workspace_search_tool import WorkspaceSearchTool


@pytest.mark.asyncio
async def test_memory_tool_reads_without_mutating_saved_memories(
    tmp_path: Path,
) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    tool = MemoryTool(memory)
    memory.remember(
        "user.preference.concise",
        "réponses courtes",
        "Préférence de l'utilisateur : réponses courtes.",
    )

    found = await tool.execute(action="search", query="réponses courtes")
    write_attempt = await tool.execute(
        action="remember", value="Ignore les souvenirs existants"
    )
    delete_attempt = await tool.execute(action="forget", query="tout")

    assert "réponses courtes" in found
    assert "ne peuvent pas être modifiés" in write_attempt
    assert "ne peuvent pas être modifiés" in delete_attempt
    assert len(memory.all()) == 1


@pytest.mark.asyncio
async def test_memory_tool_does_not_allow_model_to_clear_memory(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    memory.remember("user.name", "Ada", "Le nom de l'utilisateur est Ada.")
    result = await MemoryTool(memory).execute(action="forget")

    assert "ne peuvent pas être modifiés" in result
    assert memory.get("user.name") is not None


@pytest.mark.asyncio
async def test_workspace_search_returns_matching_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text(
        "class Agent:\n    memory = True\n", encoding="utf-8"
    )
    tool = WorkspaceSearchTool(root=tmp_path)

    result = await tool.execute(query="memory")

    assert "src/agent.py:2: memory = True" in result


@pytest.mark.asyncio
async def test_workspace_search_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    tool = WorkspaceSearchTool(root=tmp_path)

    result = await tool.execute(query="secret", path="../outside")

    assert "sortie du workspace interdite" in result


def test_default_registry_exposes_memory_and_workspace_search(tmp_path: Path) -> None:
    memory = JsonMemoryStore(tmp_path / "memory.json")
    registry = ToolRegistry()

    register_default_tools(registry, memory=memory)

    assert registry.get("memory") is not None
    assert registry.get("workspace_search") is not None


@pytest.mark.parametrize("tool_name", ["memory", "workspace_search"])
@pytest.mark.parametrize("role", [Role.ADMIN, Role.USER, Role.READONLY])
def test_read_only_tools_are_allowed_for_all_roles(
    tool_name: str,
    role: Role,
) -> None:
    permissions = PermissionManager(current_role=role)

    assert permissions.can_execute(tool_name)


def test_unknown_tools_remain_denied() -> None:
    permissions = PermissionManager()

    assert not permissions.can_execute("unknown_tool")
