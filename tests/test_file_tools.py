from pathlib import Path

import pytest

from src.infrastructure.tools.file_crud_tool import FileCrudTool
from src.infrastructure.tools.file_list_tool import FileListTool


@pytest.mark.asyncio
async def test_file_crud_create_read_update_delete(tmp_path: Path) -> None:
    tool = FileCrudTool(root=tmp_path)

    created = await tool.execute(
        action="create",
        path="notes/todo.txt",
        content="bonjour",
    )
    read = await tool.execute(action="read", path="notes/todo.txt")
    appended = await tool.execute(
        action="update",
        path="notes/todo.txt",
        content=" agent",
        mode="append",
    )
    reread = await tool.execute(action="read", path="notes/todo.txt")
    deleted = await tool.execute(action="delete", path="notes/todo.txt")

    assert created == "Fichier cree: notes/todo.txt"
    assert read == "bonjour"
    assert appended == "Fichier modifie: notes/todo.txt"
    assert reread == "bonjour agent"
    assert deleted == "Fichier supprime: notes/todo.txt"
    assert not (tmp_path / "notes" / "todo.txt").exists()


@pytest.mark.asyncio
async def test_file_crud_rejects_workspace_escape(tmp_path: Path) -> None:
    tool = FileCrudTool(root=tmp_path)

    result = await tool.execute(action="read", path="../secret.txt")

    assert result == "Chemin refuse: sortie du workspace interdite."


@pytest.mark.asyncio
async def test_file_crud_rejects_protected_files(tmp_path: Path) -> None:
    tool = FileCrudTool(root=tmp_path)

    result = await tool.execute(action="create", path=".env", content="SECRET=1")

    assert result == "Chemin refuse: fichier protege."


@pytest.mark.asyncio
async def test_file_crud_scaffolds_informal_module_request(tmp_path: Path) -> None:
    tool = FileCrudTool(root=tmp_path)
    user_input = "fais-moi le modou fichier auth meme vide stp"
    args = tool.infer_args(user_input, {"action": "", "path": "", "content": ""})

    result = await tool.execute(**args)

    assert args == {"action": "create", "path": "auth.py", "content": ""}
    assert result == "Fichier cree: auth.py"
    assert (tmp_path / "auth.py").read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_file_list_lists_workspace_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("", encoding="utf-8")
    tool = FileListTool(root=tmp_path)

    result = await tool.execute(path=".", recursive=True)

    assert "src/" in result
    assert "src/main.py" in result
    assert ".git" not in result
