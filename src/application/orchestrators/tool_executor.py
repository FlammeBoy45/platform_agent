import asyncio
import inspect
import time
from collections import defaultdict
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.protocols.event_bus import EventBus
from src.domain.protocols.logger import LoggerProtocol
from src.domain.protocols.tool import Tool, ToolProvider


class ToolExecutionError(Exception):
    pass


class ToolExecutor:
    def __init__(
        self,
        tools: ToolProvider,
        logger: LoggerProtocol,
        default_timeout: float = 5.0,
        max_retries: int = 3,
        allowed_tools: set[str] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.tools = tools
        self.logger = logger
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.allowed_tools = allowed_tools
        self.event_bus = event_bus
        self._call_times: dict[str, list[float]] = defaultdict(list)
        self._max_calls_per_minute = 30
        self._rate_limit_window = 60.0  # secondes

    def _check_rate_limit(self, tool_name: str) -> None:
        """Vérifie que le tool n'est pas appelé trop souvent."""
        now = time.time()
        # Nettoyer les vieux appels
        self._call_times[tool_name] = [
            t for t in self._call_times[tool_name] if now - t < self._rate_limit_window
        ]

        if len(self._call_times[tool_name]) >= self._max_calls_per_minute:
            raise ToolExecutionError(
                f"Rate limit atteint pour '{tool_name}' "
                f"({self._max_calls_per_minute} appels/min max). Attendez un peu."
            )

        self._call_times[tool_name].append(now)

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_input: str,
        trace_id: str | None = None,
    ) -> str:
        # 1. Vérification des Permissions
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            self.logger.warning(f"Tentative d'accès refusée : tool={tool_name}")
            raise ToolExecutionError(f"Permission refusée pour l'outil '{tool_name}'.")

        self._check_rate_limit(tool_name)

        tool = self.tools.get(tool_name)
        if not tool:
            raise ToolExecutionError(f"Outil '{tool_name}' inconnu.")

        # 2. Audit Log (Audit Trailing)
        self.logger.info(f"[AUDIT] START | tool={tool_name} | args={args}")
        start_time = time.perf_counter()

        if self.event_bus:
            await self.event_bus.emit(
                "agent.tool_called",
                {"trace_id": trace_id, "tool": tool_name, "args": args},
            )

        # Validation du schéma de base
        if not isinstance(args, dict):
            raise ToolExecutionError(
                "Arguments invalides : attendu un dictionnaire, "
                f"reçu {type(args).__name__}"
            )

        try:
            if hasattr(tool, "infer_args"):
                args = tool.infer_args(user_input, args)

                # Validation des arguments par rapport au schéma de l'outil
            optional_args = set(getattr(tool, "optional_args", ()))

            for arg_name in args.keys():
                if arg_name not in tool.args_schema:
                    raise ToolExecutionError(
                        f"Argument '{arg_name}' invalide pour l'outil '{tool_name}'."
                    )

            for req_arg, _ in tool.args_schema.items():
                # ✅ Skip les args optionnels
                if req_arg in optional_args:
                    continue
                # ✅ Vérifie que l'arg requis est présent et non vide
                if req_arg not in args or args[req_arg] is None:
                    raise ToolExecutionError(
                        f"Argument manquant '{req_arg}' pour l'outil '{tool_name}'. "
                        "Args requis: "
                        f"{[k for k in tool.args_schema if k not in optional_args]}"
                    )
            # 3. Exécution avec Retries, Timeout et Isolation
            # Ne pas réessayer les erreurs de validation des outils.
            retrier = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                retry=retry_if_not_exception_type(ToolExecutionError),
                reraise=True,
            )

            result: Any = await retrier(self._invoke_tool, tool, args)

            duration = time.perf_counter() - start_time
            self.logger.info(
                f"[AUDIT] SUCCESS | tool={tool_name} | duration={duration:.3f}s"
            )

            if self.event_bus:
                await self.event_bus.emit(
                    "agent.tool_completed",
                    {
                        "trace_id": trace_id,
                        "tool": tool_name,
                        "result": str(result),
                        "duration": duration,
                    },
                )

            return str(result)

        except TimeoutError:
            self.logger.error(f"Timeout sur l'outil {tool_name}")
            if self.event_bus:
                await self.event_bus.emit(
                    "agent.failed",
                    {"trace_id": trace_id, "tool": tool_name, "error": "timeout"},
                )
            return (
                f"Erreur : L'outil {tool_name} a dépassé le temps limite de "
                f"{self.default_timeout}s."
            )
        except Exception as e:
            self.logger.error(f"Erreur tool {tool_name}: {str(e)}")
            if self.event_bus:
                await self.event_bus.emit(
                    "agent.failed",
                    {"trace_id": trace_id, "tool": tool_name, "error": str(e)},
                )
            raise ToolExecutionError(f"Échec de l'outil '{tool_name}': {str(e)}") from e

    async def _invoke_tool(self, tool: Tool, args: dict[str, Any]) -> Any:
        """Invoque l'outil en thread séparé avec un délai maximal."""

        def wrapper() -> Any:
            # Isolation : on exécute l'outil dans son propre thread.
            # Un outil async utilise une boucle d'événements locale au thread.
            if inspect.iscoroutinefunction(tool.execute):
                return asyncio.run(tool.execute(**args))
            return tool.execute(**args)

        tool_timeout = getattr(tool, "timeout", self.default_timeout)

        # Le pool de threads évite de bloquer la boucle principale.
        return await asyncio.wait_for(asyncio.to_thread(wrapper), timeout=tool_timeout)
