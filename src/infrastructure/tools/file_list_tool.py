from pathlib import Path
from typing import Any


class FileListTool:
    name = "file_list"
    description = (
        "Liste les fichiers et dossiers dans le workspace a partir d'un chemin relatif."
    )
    args_schema = {
        "path": "Chemin relatif du dossier a lister. Defaut: .",
        "recursive": "true pour lister recursivement, false sinon.",
    }
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "liste les fichiers",
        "liste fichiers",
        "fichiers du dossier",
        "arborescence",
        "ls",
    )

    _blocked_parts = {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    _max_items = 100

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()

    async def execute(self, **kwargs: Any) -> str:
        raw_path = str(kwargs.get("path", ".") or ".").strip()
        recursive = self._as_bool(kwargs.get("recursive", False))

        try:
            path = self._safe_path(raw_path)
        except ValueError as exc:
            return str(exc)

        if not path.exists():
            return f"Liste impossible: {self._relative(path)} introuvable."
        if not path.is_dir():
            return "Liste impossible: ce chemin n'est pas un dossier."

        iterator = path.rglob("*") if recursive else path.iterdir()
        items = sorted(iterator, key=lambda item: item.as_posix())
        visible = [item for item in items if not self._is_blocked(item)]

        lines = []
        for item in visible[: self._max_items]:
            suffix = "/" if item.is_dir() else ""
            lines.append(f"{self._relative(item)}{suffix}")

        if len(visible) > self._max_items:
            lines.append(f"... {len(visible) - self._max_items} elements masques")

        return "\n".join(lines) if lines else "Aucun fichier."

    def _safe_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError("Chemin refuse: utilise un chemin relatif au workspace.")

        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Chemin refuse: sortie du workspace interdite.")

        if self._is_blocked(resolved):
            raise ValueError("Chemin refuse: dossier protege.")

        return resolved

    def _is_blocked(self, path: Path) -> bool:
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            return True
        return any(part in self._blocked_parts for part in parts)

    def _relative(self, path: Path) -> str:
        if path == self.root:
            return "."
        return path.relative_to(self.root).as_posix()

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "oui"}
