from pathlib import Path
from typing import Any


class WorkspaceSearchTool:
    name = "workspace_search"
    description = (
        "Recherche un mot ou une expression dans les fichiers texte du workspace "
        "et retourne les lignes correspondantes avec leur chemin."
    )
    args_schema = {
        "query": "Mot ou expression à rechercher.",
        "path": "Dossier relatif au workspace. Défaut: .",
    }
    optional_args = ("path",)
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "dans les fichiers",
        "dans le code",
        "dans le projet",
        "où est défini",
        "ou est defini",
        "recherche dans",
        "cherche dans",
    )
    _blocked_parts = {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
    _blocked_names = {".env", "memory.json"}
    _max_files = 2_000
    _max_file_bytes = 1_000_000
    _max_matches = 50
    _max_output_chars = 8_000

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        raw_path = str(kwargs.get("path", ".") or ".").strip()
        if not query:
            return "Recherche impossible : précise le texte à chercher."

        try:
            base = self._safe_path(raw_path)
        except ValueError as exc:
            return str(exc)
        if not base.exists():
            return f"Recherche impossible : {self._relative(base)} introuvable."
        if not base.is_dir():
            return "Recherche impossible : le chemin doit désigner un dossier."

        matches: list[str] = []
        files_seen = 0
        query_folded = query.casefold()
        try:
            for candidate in base.rglob("*"):
                if not candidate.is_file() or self._is_blocked(candidate):
                    continue
                files_seen += 1
                if files_seen > self._max_files:
                    break
                try:
                    resolved_candidate = candidate.resolve()
                    if (
                        resolved_candidate != self.root
                        and self.root not in resolved_candidate.parents
                    ):
                        continue
                    if resolved_candidate.stat().st_size > self._max_file_bytes:
                        continue
                    content = resolved_candidate.read_bytes()
                    if b"\0" in content:
                        continue
                    for line_number, line in enumerate(
                        content.decode("utf-8", errors="replace").splitlines(), start=1
                    ):
                        if query_folded in line.casefold():
                            matches.append(
                                f"{self._relative(candidate)}:{line_number}: "
                                f"{line.strip()}"
                            )
                            if len(matches) >= self._max_matches:
                                break
                except OSError:
                    continue
                if len(matches) >= self._max_matches:
                    break
        except OSError as exc:
            return f"Recherche interrompue : {exc}"

        if not matches:
            return f"Aucun résultat pour « {query} » dans {self._relative(base)}."
        result = "\n".join(matches)
        if len(result) > self._max_output_chars:
            result = result[: self._max_output_chars] + "\n…[résultat tronqué]"
        return result

    def _safe_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError("Chemin refusé : utilise un chemin relatif au workspace.")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Chemin refusé : sortie du workspace interdite.")
        if self._is_blocked(resolved):
            raise ValueError("Chemin refusé : dossier protégé.")
        return resolved

    def _is_blocked(self, path: Path) -> bool:
        try:
            relative_parts = path.relative_to(self.root).parts
        except ValueError:
            return True
        return (
            any(part in self._blocked_parts for part in relative_parts)
            or path.name in self._blocked_names
        )

    def _relative(self, path: Path) -> str:
        if path == self.root:
            return "."
        return path.relative_to(self.root).as_posix()
