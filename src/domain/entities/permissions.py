# permissions.py

from enum import Enum
from dataclasses import dataclass, field


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


@dataclass
class ToolPermission:
    """Définit les permissions d'accès à un outil."""
    tool_name: str
    allowed_roles: list[Role] = field(default_factory=lambda: [Role.ADMIN, Role.USER])
    max_args_size: int = 1000
    require_confirmation: bool = False  # Demander confirmation avant exécution


class PermissionManager:
    """Gère les permissions RBAC des outils."""

    def __init__(self, current_role: Role = Role.USER):
        self.current_role = current_role
        self._permissions: dict[str, ToolPermission] = self._default_permissions()

    def _default_permissions(self) -> dict[str, ToolPermission]:
        return {
            "echo": ToolPermission("echo", [Role.ADMIN, Role.USER, Role.READONLY]),
            "calculator": ToolPermission("calculator", [Role.ADMIN, Role.USER, Role.READONLY]),
            "datetime": ToolPermission("datetime", [Role.ADMIN, Role.USER, Role.READONLY]),
            "crypto_price": ToolPermission("crypto_price", [Role.ADMIN, Role.USER, Role.READONLY]),
            "text_stats": ToolPermission("text_stats", [Role.ADMIN, Role.USER, Role.READONLY]),
            "file_list": ToolPermission("file_list", [Role.ADMIN, Role.USER, Role.READONLY]),
            "file_crud": ToolPermission(
                "file_crud",
                [Role.ADMIN, Role.USER],
                require_confirmation=True  # Demander confirmation pour modifier des fichiers
            ),
            "python_code": ToolPermission(
                "python_code",
                [Role.ADMIN, Role.USER],  # Seul l'admin peut exécuter du code
                require_confirmation=False,
            ),
            "google_search": ToolPermission("google_search", [Role.ADMIN, Role.USER]),
            "web_fetch": ToolPermission("web_fetch", [Role.ADMIN, Role.USER]),
            "memory": ToolPermission(
                "memory",
                [Role.ADMIN, Role.USER, Role.READONLY],
            ),
            "workspace_search": ToolPermission(
                "workspace_search",
                [Role.ADMIN, Role.USER, Role.READONLY],
            ),
        }

    def can_execute(self, tool_name: str) -> bool:
        """Vérifie si le rôle actuel peut exécuter le tool."""
        perm = self._permissions.get(tool_name)
        if not perm:
            return False  # Tool inconnu → interdit
        return self.current_role in perm.allowed_roles

    def requires_confirmation(self, tool_name: str) -> bool:
        """Vérifie si le tool nécessite une confirmation utilisateur."""
        perm = self._permissions.get(tool_name)
        if not perm:
            return True  # Par défaut, demander confirmation
        return perm.require_confirmation

    def set_role(self, role: Role) -> None:
        """Change le rôle de l'utilisateur courant."""
        self.current_role = role

    def get_allowed_tools(self, all_tools: list[str]) -> list[str]:
        """Retourne la liste des tools autorisés pour le rôle courant."""
        return [t for t in all_tools if self.can_execute(t)]