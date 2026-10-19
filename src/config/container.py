from pathlib import Path

from dependency_injector import containers, providers

from src.application.orchestrators.agent_runtime import AgentRuntime
from src.config.settings import get_settings
from src.infrastructure.llm.groq_provider import GroqProvider
from src.infrastructure.llm.openai_provider import OpenAIProvider
from src.infrastructure.logging.loguru_logger import LoguruLogger
from src.infrastructure.memory.json_memory_store import JsonMemoryStore
from src.infrastructure.tools.tool_registry import ToolRegistry


def _persistent_memory_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path
    # Resolve relative paths from the repository, not the process launch folder.
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


class Container(containers.DeclarativeContainer):
    settings = providers.Singleton(get_settings)

    logger = providers.Singleton(LoguruLogger)

    tool_registry = providers.Singleton(ToolRegistry)

    memory_store = providers.Singleton(
        lambda s: JsonMemoryStore(_persistent_memory_path(s.MEMORY_PATH))
        if s.ENABLE_MEMORY else None,
        settings,
    )

    llm_provider = providers.Selector(
        settings.provided.APP_ENV,
        development=providers.Factory(
            GroqProvider,
            api_key=providers.Callable(
                lambda s: s.GROQ_API_KEY.get_secret_value(),
                settings,
            ),
            model=providers.Callable(lambda s: s.GROQ_MODEL, settings),
            logger=logger,
        ),
        testing=providers.Factory(
            GroqProvider,
            api_key=providers.Callable(
                lambda s: s.GROQ_API_KEY.get_secret_value(),
                settings,
            ),
            model=providers.Callable(lambda s: s.GROQ_MODEL, settings),
            logger=logger,
        ),
        production=providers.Factory(
            OpenAIProvider,
            api_key=providers.Callable(
                lambda s: s.OPENAI_API_KEY.get_secret_value(),
                settings,
            ),
            model=providers.Callable(lambda s: s.OPENAI_MODEL, settings),
            logger=logger,
        ),
    )

    agent_runtime = providers.Factory(
        AgentRuntime,
        llm=llm_provider,
        logger=logger,
        tools=tool_registry,
        memory=memory_store,
        memory_v2=memory_store,
        enable_memory=providers.Callable(lambda s: s.ENABLE_MEMORY, settings),
    )
