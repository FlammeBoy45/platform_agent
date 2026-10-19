from typing import Any, cast

from openai import AsyncOpenAI

from src.domain.protocols.llm_provider import LLMProvider
from src.domain.protocols.logger import LoggerProtocol


class OllamaProvider(LLMProvider):
    """Client Ollama via son API locale compatible OpenAI."""

    def __init__(
        self,
        base_url: str,
        model: str,
        logger: LoggerProtocol,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key="ollama",
        )
        self.model = model
        self.logger = logger

    async def generate(self, prompt: str) -> str:
        self.logger.debug("Ollama generate called")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.logger.debug("Ollama chat called")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[dict[str, str]]) -> Any:
        self.logger.debug("Ollama stream called")
        return await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            stream=True,
        )

    async def close(self) -> None:
        await self.client.close()
