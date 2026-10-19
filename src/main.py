import asyncio
from typing import Protocol

from src.application.orchestrators.agent_runtime import AgentRuntime
from src.config.container import Container
from src.config.logging import setup_logging
from src.infrastructure.tools.default_tools import register_default_tools


class ChatAgent(Protocol):
    async def run(self, user_input: str) -> str:
        ...


async def _conversation_loop(agent: ChatAgent) -> None:
    while True:
        try:
            user_input = input("You > ")
        except (KeyboardInterrupt, EOFError):
            print("\nAgent arrêté.")
            break

        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            response = await agent.run(user_input)
            print(f"\nAgent > {response}\n")
        except asyncio.CancelledError:
            print("\nAgent arrêté.")
            break
        except Exception as exc:
            print(f"\nError > {exc}\n")


async def main() -> None:
    setup_logging()

    container = Container()

    registry = container.tool_registry()
    register_default_tools(
        registry,
        settings=container.settings(),
        memory=container.memory_store(),
    )

    agent: AgentRuntime = container.agent_runtime()

    print("Agent ready.")
    print("Type 'exit' to quit.\n")

    await _conversation_loop(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run may re-raise Ctrl+C after the conversation loop handled it.
        pass
