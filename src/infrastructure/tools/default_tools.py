from pathlib import Path

from src.config.settings import Settings, get_settings
from src.domain.protocols.memory import LongTermMemory
from src.infrastructure.tools.calculator_tool import CalculatorTool
from src.infrastructure.tools.crypto_price_tool import CryptoPriceTool
from src.infrastructure.tools.datetime_tool import DateTimeTool
from src.infrastructure.tools.echo_tool import EchoTool
from src.infrastructure.tools.file_crud_tool import FileCrudTool
from src.infrastructure.tools.file_list_tool import FileListTool
from src.infrastructure.tools.google_search_tool import GoogleSearchTool
from src.infrastructure.tools.memory_tool import MemoryTool
from src.infrastructure.tools.python_code_tool import PythonCodeTool
from src.infrastructure.tools.text_stats_tool import TextStatsTool
from src.infrastructure.tools.tool_registry import ToolRegistry
from src.infrastructure.tools.web_fetch_tool import WebFetchTool
from src.infrastructure.tools.workspace_search_tool import WorkspaceSearchTool


def register_default_tools(
    registry: ToolRegistry,
    settings: Settings | None = None,
    memory: LongTermMemory | None = None,
) -> None:
    settings = settings or get_settings()
    workspace = Path(settings.WORKSPACE_PATH).expanduser()
    if not workspace.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        workspace = project_root / workspace
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    registry.register(EchoTool())
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(CryptoPriceTool())
    registry.register(FileCrudTool(root=workspace))
    registry.register(FileListTool(root=workspace))
    registry.register(WorkspaceSearchTool(root=workspace))
    registry.register(MemoryTool(memory))
    if settings.ENABLE_PYTHON_CODE:
        registry.register(PythonCodeTool())
    registry.register(
        GoogleSearchTool(api_key=settings.SERPAPI_API_KEY.get_secret_value())
    )
    registry.register(TextStatsTool())
    registry.register(WebFetchTool())
