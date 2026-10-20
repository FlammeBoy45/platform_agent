import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.application.memory_v2.memory_v2 import MemoryV2
from src.application.memory_v2.working_memory import WorkingMemory
from src.application.orchestrators.conversation_manager import ConversationManager
from src.application.orchestrators.evaluator import ActionEvaluator, SessionMetrics
from src.application.orchestrators.intent_classifier import (
    get_forced_tool_for_intent,
    has_action_intent,
    is_plan_resume_request,
)
from src.application.orchestrators.memory_manager import MemoryManager
from src.application.orchestrators.planner import Planner
from src.application.orchestrators.prompt_manager import PromptManager
from src.application.orchestrators.reflector import Reflector
from src.application.orchestrators.response_parser import ResponseParser
from src.application.orchestrators.runtime_errors import ToolAuthorizationError
from src.application.orchestrators.scheduler import Scheduler
from src.application.orchestrators.session_runner import run_session
from src.application.orchestrators.tool_executor import ToolExecutionError, ToolExecutor
from src.domain.entities.permissions import PermissionManager, Role
from src.domain.entities.plan import Plan, Task
from src.domain.protocols.event_bus import EventBus
from src.domain.protocols.llm_provider import LLMProvider
from src.domain.protocols.logger import LoggerProtocol
from src.domain.protocols.memory import AgentMemory, LongTermMemory
from src.domain.protocols.session_store import SessionStore
from src.domain.protocols.tool import Tool, ToolProvider

ConfirmationHandler = Callable[[str, dict[str, Any]], Awaitable[bool]]


class AgentRuntime:
    MAX_MESSAGES = 6

    def __init__(
        self,
        llm: LLMProvider,
        logger: LoggerProtocol,
        tools: ToolProvider,
        memory: LongTermMemory | None = None,
        executor: ToolExecutor | None = None,
        event_bus: EventBus | None = None,
        parser: ResponseParser | None = None,
        prompter: PromptManager | None = None,
        enable_memory: bool = True,
        memory_v2: AgentMemory | None = None,
        current_role: Role = Role.USER,
        confirmation_handler: ConfirmationHandler | None = None,
        session_store: SessionStore | None = None,
    ) -> None:

        self.llm = llm
        self.logger = logger
        self.tools = tools
        self.event_bus = event_bus
        self.session_store = session_store

        self.permissions = PermissionManager(current_role=current_role)
        self.confirmation_handler = confirmation_handler
        self.evaluator = ActionEvaluator()  # ✅

        # Core modules
        self.conversation = ConversationManager(
            self.MAX_MESSAGES,
            on_message=self._persist_conversation_message,
        )
        if self.session_store:
            self.conversation.restore_messages(
                self.session_store.load_messages(self.MAX_MESSAGES)
            )
        self.memory_manager = MemoryManager(memory)
        self.parser = parser or ResponseParser(logger)
        self.prompter = prompter or PromptManager()
        self.memory_v2: AgentMemory | None = None
        if enable_memory:
            if memory_v2 is not None:
                self.memory_v2 = memory_v2
            elif isinstance(memory, AgentMemory):
                self.memory_v2 = memory
            else:
                self.memory_v2 = MemoryV2()
        self.reflector = Reflector(self.llm)  # ✅
        self.executor = executor or ToolExecutor(tools, logger, event_bus=event_bus)

        self.scheduler = Scheduler()
        self.planner = Planner(self.llm, self.parser, memory_v2=self.memory_v2)
        self.working_memory = WorkingMemory(ttl_seconds=300)

        self.current_plan: Plan | None = None
        if self.session_store:
            saved_plan = self.session_store.load_unfinished_plan()
            if saved_plan:
                self.current_plan = Plan.model_validate(saved_plan)
        self._last_responses: list[str] = []
        self._loop_count: int = 0

    # ============================================================
    # BUILD / TOOL ALLOWED
    # ============================================================

    def _build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            *self.conversation.get_messages_as_dicts(),
        ]

    def _persist_conversation_message(self, role: str, content: str) -> None:
        if self.session_store:
            event_type = "user_message" if role == "user" else "assistant_message"
            self.session_store.append_event(event_type, {"content": content})

    def _record_session_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self.session_store:
            self.session_store.append_event(event_type, data)

    def _persist_plan_state(self) -> None:
        if self.session_store and self.current_plan:
            self._record_session_event(
                "plan_snapshot", {"plan": self.current_plan.model_dump()}
            )

    def _is_tool_allowed(self, tool: Tool, user_input: str) -> bool:
        if not tool.trigger_words:
            return True

        normalized = user_input.lower()

        if any(marker in normalized for marker in tool.trigger_words):
            return True

        if tool.name == "file_crud" and re.search(r"\.[a-z0-9]{1,5}\b", normalized):
            return True

        if tool.name == "echo" and not getattr(tool, "chat_safe", True):
            return False

        return False

    # ============================================================
    # LLM CALLS
    # ============================================================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _safe_chat(self, messages: list[dict[str, str]]) -> str:
        return await self.llm.chat(messages)

    async def _force_final_response(self, system_prompt: str, instruction: str) -> str:
        short_correction = (
            "Tu es un assistant concis et chaleureux. "
            "Réponds UNIQUEMENT avec ce format JSON:\n"
            '{"tool":"final","args":{"content":"ta réponse ici"}}\n\n'
            'Exemple: {"tool":"final","args":{"content":"Je vais bien, merci !"}}'
        )

        messages = [
            {"role": "system", "content": short_correction},
            *self.conversation.get_messages_as_dicts()[-3:],
            {"role": "user", "content": instruction},
        ]

        raw = await self._safe_chat(messages)
        action = self.parser.parse_action(raw)

        content = str(action.args.get("content", "")).strip()

        if not content or content in {"...", "null", "none"}:
            content = raw.strip()
            for prefix in ["```json", "```", "JSON:", "Réponse:"]:
                content = content.removeprefix(prefix).strip()

        if not content or len(content) < 2:
            content = instruction or "Je suis là !"

        return content

    def _needs_final_retry(self, response: str) -> bool:
        normalized = response.strip().lower()

        return (
            normalized
            in {
                "",
                "final",
                "json",
                "none",
                "null",
                "plan",
                "tool",
                "réponse directe.",
                "réponse directe",
            }
            or normalized.startswith("plan actuel")
            or normalized.startswith('{"goal"')
            or len(normalized) < 3
        )

    # ============================================================
    # LOOP DETECTION
    # ============================================================

    def _is_loop_detected(self, response: str, user_input: str) -> bool:
        """Détecte si le LLM tourne en boucle sur une réponse inutile."""
        if (
            self._last_responses
            and response.strip() == self._last_responses[-1].strip()
        ):
            self._loop_count += 1
        else:
            self._loop_count = 0

        self._last_responses.append(response)
        if len(self._last_responses) > 5:
            self._last_responses = self._last_responses[-5:]

        if self._loop_count >= 2:
            self.logger.warning(f"Boucle détectée: réponse répétée {self._loop_count}x")
            return True

        return False

    def _has_unfinished_plan(self) -> bool:
        return bool(
            self.current_plan
            and any(
                task.status in {"pending", "in_progress"}
                or (task.status == "failed" and task.attempts < task.max_attempts)
                for task in self.current_plan.tasks
            )
        )

    def _set_task_result(self, task: Task, success: bool, result: str) -> None:
        task.result = result[:2_000]
        if success:
            task.status = "completed"
        else:
            task.attempts += 1
            task.status = "failed"
        self._persist_plan_state()

    @staticmethod
    def _tool_result_succeeded(result: str) -> bool:
        normalized = result.casefold()
        failure_markers = (
            "erreur",
            "échec",
            "echec",
            "impossible",
            "refus",
            "refusé",
            "refusee",
            "introuvable",
            "invalide",
            "annulé",
            "annule",
            "timeout",
            "code sortie: 1",
            "failed",
            "error",
            "denied",
            "not found",
            "timezone inconnue",
            "url manquante",
            "creation refusee",
            "création refusée",
            "modification impossible",
            "modification refusee",
            "suppression impossible",
            "suppression refusee",
            "lecture impossible",
            "lecture refusee",
            "liste impossible",
            "chemin refuse",
            "chemin refusé",
            "action invalide",
            "mode invalide",
            "url refusee",
            "recherche impossible",
            "recherche interrompue",
            "accès refusé",
            "acces refuse",
            "permission refusée",
            "permission refusee",
        )
        normalized = normalized.strip()
        return bool(normalized) and not normalized.startswith(failure_markers)

    def _merge_plan_update(self, proposed: Plan) -> Plan:
        current = self.current_plan
        if current is None or not self._has_unfinished_plan():
            return proposed
        if proposed.goal.strip().casefold() != current.goal.strip().casefold():
            return current

        current_tasks = {
            task.description.strip().casefold(): task for task in current.tasks
        }
        merged_tasks: list[Task] = []
        proposed_descriptions: set[str] = set()
        for task in proposed.tasks:
            key = task.description.strip().casefold()
            proposed_descriptions.add(key)
            existing = current_tasks.get(key)
            if existing and existing.status in {"completed", "in_progress", "failed"}:
                merged_tasks.append(existing)
            else:
                merged_tasks.append(task)

        merged_tasks.extend(
            task
            for task in current.tasks
            if task.description.strip().casefold() not in proposed_descriptions
        )
        proposed.tasks = merged_tasks
        return proposed

    @staticmethod
    def _is_plan_resume_request(user_input: str) -> bool:
        return is_plan_resume_request(user_input)

    def _has_action_intent(self, user_input: str) -> bool:
        return has_action_intent(user_input)

    def _get_forced_tool_for_intent(
        self, user_input: str
    ) -> tuple[str, dict[str, Any]]:
        return get_forced_tool_for_intent(user_input)

    # ============================================================
    # HELPERS
    # ============================================================

    def _answer_tools_list_if_requested(self, user_input: str) -> str | None:
        normalized = user_input.lower()

        if "tool" in normalized and any(
            x in normalized for x in ["liste", "disponible", "quels"]
        ):
            return "\n".join(
                ["Tools disponibles :"]
                + [f"- {t.name}: {t.description}" for t in self.tools.list_tools()]
            )

        return None

    def _answer_memory_list_if_requested(self, user_input: str) -> str | None:
        normalized = user_input.casefold()
        memory_phrases = (
            "montre ma mémoire",
            "montre ta mémoire",
            "montre-moi ma mémoire",
            "montre-moi ta mémoire",
            "affiche ma mémoire",
            "affiche ta mémoire",
            "affiche-moi ma mémoire",
            "affiche-moi ta mémoire",
            "affiche moi la mémoire",
            "qu'as-tu retenu",
            "qu’est-ce que tu as retenu",
            "qu'est-ce que tu as retenu",
            "qu'est-ce que tu sais sur moi",
            "qu’est-ce que tu sais sur moi",
            "que sais-tu sur moi",
            "montre les souvenirs",
            "affiche les souvenirs",
        )
        if not any(phrase in normalized for phrase in memory_phrases):
            return None

        memory = self.memory_manager.memory
        if memory is None or not hasattr(memory, "list_recent"):
            return "La mémoire persistante n'est pas activée."

        facts = memory.list_recent(limit=20)
        if not facts:
            return "Je n'ai encore rien mémorisé."

        entries = ["Voici mes souvenirs récents :"]
        entries.extend(f"- {' '.join(fact.text.split())}" for fact in facts)
        return "\n".join(entries)

    async def _run_tool(
        self, tool: Tool, args: dict[str, Any], user_input: str, trace_id: str
    ) -> str:
        authorization_args = args
        if hasattr(tool, "infer_args"):
            authorization_args = tool.infer_args(user_input, args)
        if not self.permissions.can_execute(tool.name, authorization_args):
            raise ToolAuthorizationError(
                f"Accès refusé : vous n'avez pas la permission d'utiliser "
                f"'{tool.name}'. Rôle actuel: {self.permissions.current_role.value}"
            )

        if self.permissions.requires_confirmation(tool.name, authorization_args):
            if self.confirmation_handler is None:
                raise ToolAuthorizationError(
                    f"Action '{tool.name}' bloquée : une confirmation utilisateur "
                    "est requise, mais aucun gestionnaire de confirmation "
                    "n'est configuré."
                )
            if not await self.confirmation_handler(tool.name, authorization_args):
                raise ToolAuthorizationError(
                    f"Action '{tool.name}' annulée : confirmation refusée."
                )

        try:
            self._record_session_event(
                "tool_started",
                {
                    "trace_id": trace_id,
                    "tool": tool.name,
                },
            )
            start = time.time()
            result = await self.executor.execute(
                tool.name, args, user_input, trace_id=trace_id
            )
            duration = (time.time() - start) * 1000
            self._record_session_event(
                "tool_finished",
                {
                    "trace_id": trace_id,
                    "tool": tool.name,
                    "duration_ms": round(duration, 2),
                    "result": result[:2_000],
                },
            )
            return result
        except ToolExecutionError as e:
            msg = f"ERREUR TOOL : {e}"
            self._record_session_event(
                "tool_failed",
                {
                    "trace_id": trace_id,
                    "tool": tool.name,
                    "error": msg[:2_000],
                },
            )
            if self.event_bus:
                await self.event_bus.emit(
                    "agent.error", {"trace_id": trace_id, "error": msg}
                )
            return msg

    # ============================================================
    # MAIN RUN
    # ============================================================

    async def run(self, user_input: str, max_steps: int = 10) -> str:
        trace_id = str(uuid.uuid4())
        metrics = self.evaluator.start_session(trace_id, user_input)
        response: str | None = None
        error: BaseException | None = None
        self.logger.info(f"Autonomous session started | trace_id={trace_id}")

        try:
            self._record_session_event("turn_started", {"trace_id": trace_id})
            if self.event_bus:
                await self.event_bus.emit(
                    "agent.started",
                    {"trace_id": trace_id, "user_input": user_input},
                )
            response = await self._run_session(
                user_input,
                max_steps,
                trace_id,
                metrics,
            )
            return response
        except asyncio.CancelledError as exc:
            metrics.status = "cancelled"
            error = exc
            raise
        except Exception as exc:
            metrics.status = "failed"
            error = exc
            raise
        finally:
            if metrics.status == "running":
                metrics.status = "completed"
            self.evaluator.end_session(metrics)
            self._record_session_event(
                "turn_finished",
                {
                    "trace_id": trace_id,
                    "status": metrics.status,
                    "metrics": {
                        "actions": metrics.total_actions,
                        "successful_actions": metrics.successful_actions,
                        "failed_actions": metrics.failed_actions,
                        "success_rate": metrics.success_rate,
                        "retries": metrics.retries,
                        "replans": metrics.replans,
                        "duration_ms": round(metrics.session_duration_ms, 2),
                    },
                    "response": response[:2_000] if response else None,
                    "error": str(error)[:2_000] if error else None,
                },
            )
            if self._has_unfinished_plan():
                self._persist_plan_state()
            if self.event_bus:
                event_type = (
                    "agent.completed"
                    if metrics.status == "completed"
                    else "agent.failed"
                    if metrics.status == "failed"
                    else "agent.interrupted"
                )
                event_data: dict[str, Any] = {
                    "trace_id": trace_id,
                    "status": metrics.status,
                    "response": response,
                    "metrics": metrics.summary(),
                    "duration_ms": metrics.session_duration_ms,
                }
                if error is not None:
                    event_data["error"] = str(error)
                await self.event_bus.emit(event_type, event_data)

    async def _run_session(
        self,
        user_input: str,
        max_steps: int,
        trace_id: str,
        metrics: SessionMetrics,
    ) -> str:
        return await run_session(self, user_input, max_steps, trace_id, metrics)

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> None:
        self.conversation.reset()
        if self.session_store:
            self.session_store.new_session()
        self.current_plan = None
        self._last_responses = []
        self._loop_count = 0
        self._retry_count = 0
        self.working_memory.clear()
        self.logger.info("Agent reset")
