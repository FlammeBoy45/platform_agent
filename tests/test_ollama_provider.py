from typing import Any

import pytest

import src.infrastructure.llm.ollama_provider as ollama_provider_module
from src.config.container import Container
from src.config.environments import Environment
from src.config.settings import Settings
from src.domain.protocols.logger import LoggerProtocol
from src.infrastructure.llm.ollama_provider import OllamaProvider


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


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return "stream"
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": "réponse locale"},
                            )(),
                        },
                    )(),
                ],
            },
        )()


class FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs
        self.completions = FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ollama_provider_uses_openai_compatible_local_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[FakeOpenAI] = []

    def create_client(**kwargs: Any) -> FakeOpenAI:
        client = FakeOpenAI(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(ollama_provider_module, "AsyncOpenAI", create_client)
    provider = OllamaProvider(
        base_url="http://localhost:11434/v1/",
        model="llama3.2",
        logger=NullLogger(),
    )

    response = await provider.chat([{"role": "user", "content": "Bonjour"}])
    assert response == "réponse locale"
    assert clients[0].options == {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
    assert clients[0].completions.calls == [
        {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Bonjour"}],
        }
    ]

    streamed = await provider.stream([{"role": "user", "content": "Bonjour"}])
    assert streamed == "stream"
    await provider.close()
    assert clients[0].closed


def test_ollama_environment_selects_local_provider() -> None:
    container = Container()
    container.settings.override(
        Settings(
            APP_ENV=Environment.OLLAMA,
            OLLAMA_BASE_URL="http://127.0.0.1:11434/v1",
            OLLAMA_MODEL="qwen2.5",
        )
    )
    try:
        provider = container.llm_provider()
    finally:
        container.settings.reset_override()

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5"
    assert provider.client.base_url == "http://127.0.0.1:11434/v1/"
