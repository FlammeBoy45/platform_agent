from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str
    args_schema: dict[str, str]
    return_direct: bool
    trigger_words: tuple[str, ...]

    async def execute(self, **kwargs: Any) -> Any: ...


class ToolProvider(Protocol):
    def get(self, name: str) -> Tool | None: ...

    def list_tools(self) -> list[Tool]: ...
