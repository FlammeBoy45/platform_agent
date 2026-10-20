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
from src.application.orchestrators.memory_manager import MemoryManager
from src.application.orchestrators.planner import Planner
from src.application.orchestrators.prompt_manager import PromptManager
from src.application.orchestrators.reflector import Reflector
from src.application.orchestrators.response_parser import ResponseParser
from src.application.orchestrators.scheduler import Scheduler
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


class ToolAuthorizationError(Exception):
    """Raised when a tool call is unauthorized or lacks user confirmation."""


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
        normalized = re.sub(r"[^\w\s]", " ", user_input.casefold())
        normalized = " ".join(normalized.split())
        resume_phrases = (
            "continue",
            "continuer",
            "continue la tâche",
            "continue la tache",
            "poursuis",
            "poursuivre",
            "reprends",
            "reprendre",
            "reprends la tâche",
            "reprends la tache",
            "termine",
            "termine la tâche",
            "termine la tache",
            "fais la suite",
            "fais la suite du plan",
            "où en étais tu",
            "ou en etais tu",
        )
        return any(
            normalized == phrase or normalized.startswith(f"{phrase} ")
            for phrase in resume_phrases
        )

    def _has_action_intent(self, user_input: str) -> bool:
        """Détecte si l'utilisateur demande une action."""
        action_keywords = [
            "créer",
            "crée",
            "cree",
            "creer",
            "fais",
            "faire",
            "prépare",
            "préparer",
            "prepare",
            "preparer",
            "ponds",
            "implémente",
            "implemente",
            "supprimer",
            "supprime",
            "effacer",
            "efface",
            "modifier",
            "modifie",
            "ajouter",
            "ajoute",
            "écrire",
            "ecrire",
            "lis",
            "lire",
            "lit",
            "déplacer",
            "copier",
            "calcule",
            "cherche",
            "recherche",
            "google",
            "internet",
            "web",
            "prix",
            "cours",
            "valeur",
            "script",
            "code",
            "python",
            "programme",
            "fonction",
            "algorithme",
            "heure",
            "date",
            "aujourd'hui",
            "demain",
        ]
        ui = user_input.lower()
        return any(kw in ui for kw in action_keywords)

    def _get_forced_tool_for_intent(
        self, user_input: str
    ) -> tuple[str, dict[str, Any]]:
        """Retourne le tool et les args forcés selon l'intention détectée."""
        ui = user_input.lower()

        file_targets = (
            "fichier",
            "file",
            "module",
            "modou",
            "dossier",
            "folder",
            "répertoire",
            "repertoire",
        )
        file_actions = (
            "crée",
            "créer",
            "cree",
            "creer",
            "fais",
            "faire",
            "prépare",
            "préparer",
            "prepare",
            "preparer",
            "ajoute",
            "ajouter",
            "écris",
            "ecris",
            "écrire",
            "ecrire",
            "lis",
            "lire",
            "ouvre",
            "ouvrir",
            "modifie",
            "modifier",
            "supprime",
            "supprimer",
            "efface",
            "effacer",
        )
        if any(word in ui for word in file_targets) and any(
            word in ui for word in file_actions
        ):
            return ("file_crud", {"action": "", "path": "", "content": ""})
        if re.search(r"\.[a-z0-9]{1,8}\b", ui) and any(
            word in ui for word in file_actions
        ):
            return ("file_crud", {"action": "", "path": "", "content": ""})

        calc_keywords = [
            "calcule",
            "calcul",
            "combien fait",
            "additionne",
            "multiplie",
            "divise",
            "soustrait",
            "pourcentage",
        ]
        if any(kw in ui for kw in calc_keywords):
            return ("calculator", {"expression": user_input})

        crypto_keywords = [
            "bitcoin",
            "ethereum",
            "crypto",
            "prix",
            "cours",
            "btc",
            "eth",
            "valeur",
            "acheter",
            "vendre",
        ]  # ✅
        if any(kw in ui for kw in crypto_keywords):
            return ("crypto_price", {"symbol": "BTC"})

        time_keywords = [
            "heure",
            "date",
            "aujourd'hui",
            "demain",
            "quel jour",
            "quel jour de",
            "quel mois",
            "quelle année",
            "time",
            "day",
            "month",
            "year",
        ]
        if any(kw in ui for kw in time_keywords):
            offset = re.search(r"\butc\s*([+-]\s*\d{1,2}(?::?\d{2})?)", ui)
            timezone = f"UTC{offset.group(1).replace(' ', '')}" if offset else "UTC"
            return ("datetime", {"timezone": timezone})

        search_keywords = ["cherche", "recherche", "google", "internet", "web"]
        if any(kw in ui for kw in search_keywords):
            if any(
                phrase in ui
                for phrase in (
                    "dans le code",
                    "dans les fichiers",
                    "dans le projet",
                    "où est défini",
                    "ou est defini",
                )
            ):
                return ("workspace_search", {"query": user_input, "path": "."})
            return ("google_search", {"query": user_input})

        code_keywords = [
            "script",
            "code",
            "python",
            "programme",
            "fonction",
            "algorithme",
        ]
        if any(kw in ui for kw in code_keywords):
            return ("python_code", {"code": user_input})

        return ("final", {"content": f"Action demandée: {user_input}"})

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

        # Keep unfinished plans in this runtime; only explicit follow-ups resume them.
        self._retry_count = 0
        self._last_responses = []
        self._loop_count = 0

        previous_user_input = next(
            (
                message["content"]
                for message in reversed(self.conversation.get_messages_as_dicts())
                if message["role"] == "user"
            ),
            None,
        )
        self.conversation.add_user_message(user_input)

        forget_result = self.memory_manager.forget_requested_memory(user_input)
        if forget_result is not None:
            _, removed = forget_result
            if self.memory_manager.memory is None:
                response = "La mémoire persistante est désactivée."
            else:
                response = (
                    "J'ai oublié le souvenir demandé."
                    if removed
                    else "Je n'ai trouvé aucun souvenir correspondant."
                )
            self.conversation.add_assistant_message(response)
            return response

        remembered_fact = self.memory_manager.learn_facts(user_input)
        save_memory_request = self.memory_manager.is_save_memory_request(user_input)
        if remembered_fact or save_memory_request:
            if self.memory_manager.memory is None:
                response = (
                    "La mémoire persistante est désactivée; "
                    "je ne peux pas l'enregistrer."
                )
            elif remembered_fact:
                response = "C'est noté, je l'ai enregistré dans ma mémoire persistante."
            else:
                saved_fact = self.memory_manager.save_previous_statement(
                    user_input, previous_user_input
                )
                response = (
                    "C'est noté, je l'ai enregistré dans ma mémoire persistante."
                    if saved_fact
                    else "Quel souvenir veux-tu que j'enregistre ?"
                )

            self.conversation.add_assistant_message(response)
            if self.memory_v2:
                self.memory_v2.store_episode(
                    user_input, {"type": "memory_write", "output": response}
                )
            return response

        memory_response = self._answer_memory_list_if_requested(user_input)
        if memory_response:
            self.conversation.add_assistant_message(memory_response)
            return memory_response

        # TOOL SHORTCUT
        tools_response = self._answer_tools_list_if_requested(user_input)
        if tools_response:
            self.conversation.add_assistant_message(tools_response)
            if self.memory_v2:
                self.memory_v2.store_episode(
                    user_input, {"type": "tool_list", "output": tools_response}
                )
            return tools_response

        # MEMORY SHORTCUT
        memory_response = self.memory_manager.answer_from_memory(user_input)
        if memory_response:
            self.conversation.add_assistant_message(memory_response)
            if self.memory_v2:
                self.memory_v2.store_episode(
                    user_input, {"type": "memory_hit", "output": memory_response}
                )
            return memory_response

        # PLAN INIT
        resume_plan = self._has_unfinished_plan() and self._is_plan_resume_request(
            user_input
        )
        if resume_plan and self.current_plan:
            for pending_task in self.current_plan.tasks:
                if pending_task.status == "in_progress" or (
                    pending_task.status == "failed"
                    and pending_task.attempts < pending_task.max_attempts
                ):
                    pending_task.status = "pending"
            self._persist_plan_state()
        else:
            self.current_plan = await self.planner.create_plan(user_input)
            self._persist_plan_state()

        last_response = ""

        for _ in range(max_steps):
            # TASK PICK
            task = (
                self.scheduler.get_next_task(self.current_plan)
                if self.current_plan
                else None
            )
            task_query = task.description if task else user_input

            # MEMORY RETRIEVAL
            retrieved = self.memory_v2.retrieve(task_query) if self.memory_v2 else []
            memory_context = "\n".join(
                f"- {getattr(m, 'content', getattr(m, 'text', str(m)))}"
                for m in retrieved[:5]
            )

            # =========================
            # PROMPT BUILD
            # =========================
            system_prompt = self.prompter.build_system_prompt(
                self.tools,
                self.memory_manager.memory,
                user_input,
                plan=self.current_plan,
            )

            if memory_context:
                system_prompt += f"\n\nMEMORY V2:\n{memory_context}"

            # ✅ WORKING MEMORY
            working_context = self.working_memory.get_context_for_prompt()
            if working_context:
                system_prompt += working_context

            if task:
                system_prompt += f"\n\nTACHE ACTIVE:\n{task.description}"
                if task.status == "pending":
                    task.status = "in_progress"
                    self._persist_plan_state()

            # =========================
            # LLM CALL
            # =========================
            raw = await self._safe_chat(self._build_messages(system_prompt))
            action = self.parser.parse_action(raw)

            # =========================
            # ✅ REFUSER FINAL SI ACTION DEMANDÉE
            # =========================
            if action.tool == "final" and self._has_action_intent(user_input):
                response_text = str(action.args.get("content", ""))
                forced_tool, forced_args = self._get_forced_tool_for_intent(user_input)

                if forced_tool != "final" and self.tools.get(forced_tool):
                    self.logger.warning(
                        f"Action demandée mais LLM répond final: "
                        f"'{response_text[:80]}...' → Forcer tool."
                    )
                    action.tool = forced_tool
                    action.args = forced_args
                    self.logger.info(
                        f"Tool forcé: {forced_tool} avec args {forced_args}"
                    )

            # =========================
            # ANTI-ARGS-VIDE
            # =========================
            if action.tool and action.tool != "final":
                tool_ref = self.tools.get(action.tool)
                infer_args = getattr(tool_ref, "infer_args", None)
                if callable(infer_args):
                    action.args = infer_args(user_input, action.args)

            if (
                action.tool
                and action.tool != "final"
                and self.tools.get(action.tool)
                and (not action.args or action.args == {})
            ):
                self.logger.warning(f"Args vides pour '{action.tool}'. Correction...")
                tool_info = ""
                tool_ref = self.tools.get(action.tool)
                if tool_ref:
                    args_schema = ", ".join(
                        f'"{k}": "{v}"' for k, v in tool_ref.args_schema.items()
                    )
                    tool_info = (
                        f"L'outil à utiliser est OBLIGATOIREMENT '{action.tool}'. "
                        f'Format requis : {{"tool":"{action.tool}", '
                        f'"args":{{{args_schema}}}, '
                        f'"plan":{{"goal":"...", "tasks":'
                        f'[{{"description":"...", "status":"pending"}}]}}}}'
                    )

                correction_prompt = (
                    f"Tu as choisi l'outil '{action.tool}' mais avec des args vides.\n"
                    f"Corrige UNIQUEMENT les args, ne change PAS d'outil.\n\n"
                    f"{tool_info}\n\n"
                    f'Requête utilisateur : "{user_input}"\n\n'
                    f"Réponds UNIQUEMENT en JSON correct."
                )
                try:
                    raw = await self._safe_chat(
                        [
                            {"role": "system", "content": correction_prompt},
                        ]
                    )
                    action = self.parser.parse_action(raw)
                except Exception:
                    pass

                if action.tool != "final" and (not action.args or action.args == {}):
                    self.logger.error(
                        f"Échec correction args pour '{action.tool}', fallback final"
                    )
                    action.tool = "final"
                    action.args = {
                        "content": (
                            f"Je n'ai pas réussi à utiliser {action.tool}. "
                            "Peux-tu reformuler ?"
                        )
                    }

            # =========================
            # DÉTECTEUR DE BOUCLE
            # =========================
            if action.tool == "final":
                response_text = str(action.args.get("content", ""))
                if self._is_loop_detected(
                    response_text, user_input
                ) and self._has_action_intent(user_input):
                    forced_tool, forced_args = self._get_forced_tool_for_intent(
                        user_input
                    )
                    self.logger.warning(f"Boucle détectée → force '{forced_tool}'")
                    action.tool = forced_tool
                    action.args = forced_args

            if action.plan:
                for proposed_task in action.plan.tasks:
                    proposed_task.status = "pending"
                    proposed_task.result = None
                    proposed_task.attempts = 0
                proposed_plan = self._merge_plan_update(action.plan)
                validated_plan = self.planner.validate_plan(proposed_plan)
                if validated_plan is None:
                    self.logger.warning(
                        "Plan du LLM ignoré : dépendance inconnue, doublon, "
                        "cycle ou trop d'étapes."
                    )
                else:
                    self.current_plan = validated_plan
                    self._persist_plan_state()

            tool_name = str(action.tool or "final").strip().casefold()
            args = action.args or {}

            # =========================
            # FINAL RESPONSE
            # =========================
            if tool_name == "final":
                response = str(args.get("content", ""))

                if self._needs_final_retry(response):
                    response = await self._force_final_response(
                        system_prompt, "Réponds naturellement et brièvement."
                    )

                if task:
                    forced_tool, _ = self._get_forced_tool_for_intent(
                        f"{user_input} {task.description}"
                    )
                    if (
                        self._has_action_intent(f"{user_input} {task.description}")
                        and forced_tool != "final"
                        and self.tools.get(forced_tool)
                    ):
                        task.status = "in_progress"
                        task.result = "Aucune réussite d'outil confirmée."
                        response = (
                            "Je ne peux pas confirmer que l'action a été exécutée; "
                            "je la laisse non terminée."
                        )
                        self._persist_plan_state()
                        self.conversation.add_assistant_message(response)
                        return response
                    self._set_task_result(task, True, response)

                self.conversation.add_assistant_message(response)
                if self.memory_v2:
                    self.memory_v2.store_episode(
                        user_input,
                        {
                            "type": "final",
                            "output": response,
                            "task": task.description if task else None,
                        },
                    )

                if (
                    task
                    and self.current_plan
                    and self.scheduler.get_next_task(self.current_plan)
                ):
                    last_response = response
                    continue

                return response

            # =========================
            # EMPTY TOOL FIX
            # =========================
            if tool_name in {"", "none", "null"}:
                response = await self._force_final_response(
                    system_prompt, "Réponds directement à l'utilisateur."
                )
                if self.memory_v2:
                    self.memory_v2.store_episode(
                        user_input,
                        {
                            "type": "direct",
                            "output": response,
                        },
                    )
                self.conversation.add_assistant_message(response)
                return response

            # =========================
            # TOOL RESOLUTION
            # =========================
            tool = self.tools.get(tool_name)
            success = False
            observation = ""

            if not tool:
                self.logger.warning(
                    f"Tool inconnu: '{tool_name}'. "
                    f"Disponibles: {[t.name for t in self.tools.list_tools()]}"
                )
                available_tools = ", ".join(
                    tool.name for tool in self.tools.list_tools()
                )
                observation = (
                    f'Je n\'ai pas acces au tool "{tool_name}". '
                    f"Tools disponibles: {available_tools}."
                )
                self.conversation.add_assistant_message(observation)
                return observation

            elif not self._is_tool_allowed(tool, user_input):
                observation = await self._force_final_response(
                    system_prompt, f"Réponds à: {user_input}"
                )
                success = False

            else:
                try:
                    start_time = time.time()
                    observation = await self._run_tool(tool, args, user_input, trace_id)
                    duration_ms = (time.time() - start_time) * 1000
                    success = self._tool_result_succeeded(observation)

                    # ✅ Enregistrer métrique
                    self.evaluator.record_action(
                        metrics, tool_name, success, duration_ms, observation
                    )
                except ToolAuthorizationError as e:
                    response = str(e)
                    metrics.status = "failed"
                    self.evaluator.record_action(
                        metrics,
                        tool_name,
                        False,
                        0,
                        response,
                        error=response,
                    )
                    self.logger.warning(response)
                    self.conversation.add_assistant_message(response)
                    if self.event_bus:
                        await self.event_bus.emit(
                            "agent.authorization_denied",
                            {
                                "trace_id": trace_id,
                                "tool": tool_name,
                                "reason": response,
                            },
                        )
                    return response
                except Exception as e:
                    observation = f"ERROR: {str(e)}"
                    success = False
                    self.evaluator.record_action(
                        metrics, tool_name, False, 0, observation, error=str(e)
                    )

            # =========================
            # ✅ RÉFLEXION (skip si return_direct + succès)
            # =========================
            goal = task.description if task else user_input

            # Ne pas réfléchir sur les tools à retour direct qui ont réussi
            skip_reflection = (
                tool and tool.return_direct and success and "ERREUR" not in observation
            )

            if skip_reflection:
                evaluation = {
                    "status": "ok",
                    "analysis": "Succès direct",
                    "suggestion": "",
                }
            else:
                evaluation = await self.reflector.evaluate(
                    goal=goal,
                    action_taken=f"{tool_name}({args})",
                    result=observation,
                    success=success,
                )

            if evaluation["status"] == "retry":
                self.evaluator.record_retry(metrics)
            elif evaluation["status"] == "replan":
                self.evaluator.record_replan(metrics)

            self.logger.info(
                f"[REFLECTION] status={evaluation['status']} | "
                f"analysis={evaluation['analysis'][:100]}"
            )

            # =========================
            # ✅ REPLANIFICATION (limité à 1 retry)
            # =========================
            retry_count = getattr(self, "_retry_count", 0)

            if evaluation["status"] in {"replan", "retry"} and self.current_plan:
                if retry_count >= 1:
                    self.logger.warning(f"Max retries atteint ({retry_count}), abandon")
                    response = (
                        f"Je n'ai pas réussi après {retry_count} tentative(s). "
                        f"Dernier résultat: {observation}"
                    )
                    self.conversation.add_assistant_message(response)
                    if task:
                        self._set_task_result(task, False, observation)
                    metrics.status = "failed"
                    return response

                self._retry_count = retry_count + 1
                suggestion = evaluation.get("suggestion", "")
                self.logger.warning(
                    f"Replanification #{retry_count + 1}: "
                    f"{evaluation['status']} - {suggestion}"
                )

                if evaluation["status"] == "retry":
                    system_prompt += (
                        f"\n\n⚠️ RÉFLEXION : {evaluation['analysis']}\n"
                        f"SUGGESTION : {suggestion}\n"
                        f"Réessaie avec cette approche."
                    )
                    continue

                elif evaluation["status"] == "replan":
                    self.current_plan = await self.planner.create_plan(
                        f"{goal} (échec précédent: {evaluation['analysis']})"
                    )
                    continue

            # =========================
            # GIVE UP
            # =========================
            if evaluation["status"] == "give_up":
                response = (
                    f"Je n'ai pas réussi à accomplir cette tâche. "
                    f"Raison : {evaluation['analysis']}"
                )
                self.conversation.add_assistant_message(response)
                if task:
                    self._set_task_result(task, False, evaluation["analysis"])
                metrics.status = "failed"
                return response

            # TASK UPDATE
            if task:
                self._set_task_result(task, success, observation)

            # =========================
            # SKILL MEMORY
            # =========================
            if success and tool and self.memory_v2:
                self.memory_v2.store_skill(
                    task=tool_name,
                    steps=[task.description if task else tool_name, tool_name],
                    success=True,
                )
                # ✅ WORKING MEMORY
                self.working_memory.put(f"resultat_{tool_name}", str(observation)[:200])
            # EPISODIC MEMORY
            if self.memory_v2:
                self.memory_v2.store_episode(
                    user_input,
                    {
                        "type": "tool_execution",
                        "tool": tool_name,
                        "output": observation,
                        "success": success,
                    },
                )

            # RETURN DIRECT TOOL
            if tool and tool.return_direct:
                if (
                    task
                    and self.current_plan
                    and self.scheduler.get_next_task(self.current_plan)
                ):
                    self.conversation.add_assistant_message(observation)
                    last_response = observation
                    continue
                if not success:
                    metrics.status = "failed"
                self.conversation.add_assistant_message(observation)
                return observation

            self.conversation.add_assistant_message(observation)
            last_response = observation

        metrics.status = "incomplete"
        response = f"Max steps atteints. Dernier état: {last_response}"
        self.conversation.add_assistant_message(response)
        return response

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
