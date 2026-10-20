import asyncio
from typing import Protocol

from src import __version__
from src.application.orchestrators.agent_runtime import AgentRuntime
from src.config.container import Container
from src.config.logging import setup_logging
from src.infrastructure.tools.default_tools import register_default_tools


class ChatAgent(Protocol):
    async def run(self, user_input: str) -> str: ...


async def _confirm_tool_action(tool_name: str, args: dict[str, object]) -> bool:
    action = str(args.get("action", "exécuter"))
    path = str(args.get("path", "")).strip()
    target = f" sur « {path} »" if path else ""
    prompt = f"Confirmer l'action {action}{target} avec l'outil {tool_name} ? [o/N] "
    try:
        answer = await asyncio.to_thread(input, prompt)
    except EOFError:
        return False
    return answer.strip().casefold() in {"o", "oui", "y", "yes"}


async def _conversation_loop(agent: ChatAgent) -> None:
    while True:
        try:
            user_input = input("You > ")
        except (KeyboardInterrupt, EOFError):
            print("\nAgent arrêté.")
            break

        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.strip().casefold() == "/version":
            print(f"Agent Platform v{__version__}\n")
            continue
        if user_input.strip().lower() == "/new":
            reset = getattr(agent, "reset", None)
            if callable(reset):
                reset()
            print("Nouvelle session.\n")
            continue

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

    agent: AgentRuntime = container.agent_runtime(
        confirmation_handler=_confirm_tool_action
    )

    print("Agent ready.")
    print("Type '/version', '/new' or 'exit'.\n")

    await _conversation_loop(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run may re-raise Ctrl+C after the conversation loop handled it.
        pass
