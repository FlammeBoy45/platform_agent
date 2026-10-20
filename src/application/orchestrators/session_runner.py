from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.application.orchestrators.evaluator import SessionMetrics
from src.application.orchestrators.runtime_errors import ToolAuthorizationError

if TYPE_CHECKING:
    from src.application.orchestrators.agent_runtime import AgentRuntime


async def run_session(
    runtime: AgentRuntime,
    user_input: str,
    max_steps: int,
    trace_id: str,
    metrics: SessionMetrics,
) -> str:

    # Keep unfinished plans in this runtime; only explicit follow-ups resume them.
    runtime._retry_count = 0
    runtime._last_responses = []
    runtime._loop_count = 0

    previous_user_input = next(
        (
            message["content"]
            for message in reversed(runtime.conversation.get_messages_as_dicts())
            if message["role"] == "user"
        ),
        None,
    )
    runtime.conversation.add_user_message(user_input)

    forget_result = runtime.memory_manager.forget_requested_memory(user_input)
    if forget_result is not None:
        _, removed = forget_result
        if runtime.memory_manager.memory is None:
            response = "La mémoire persistante est désactivée."
        else:
            response = (
                "J'ai oublié le souvenir demandé."
                if removed
                else "Je n'ai trouvé aucun souvenir correspondant."
            )
        runtime.conversation.add_assistant_message(response)
        return response

    remembered_fact = runtime.memory_manager.learn_facts(user_input)
    save_memory_request = runtime.memory_manager.is_save_memory_request(user_input)
    if remembered_fact or save_memory_request:
        if runtime.memory_manager.memory is None:
            response = (
                "La mémoire persistante est désactivée; je ne peux pas l'enregistrer."
            )
        elif remembered_fact:
            response = "C'est noté, je l'ai enregistré dans ma mémoire persistante."
        else:
            saved_fact = runtime.memory_manager.save_previous_statement(
                user_input, previous_user_input
            )
            response = (
                "C'est noté, je l'ai enregistré dans ma mémoire persistante."
                if saved_fact
                else "Quel souvenir veux-tu que j'enregistre ?"
            )

        runtime.conversation.add_assistant_message(response)
        if runtime.memory_v2:
            runtime.memory_v2.store_episode(
                user_input, {"type": "memory_write", "output": response}
            )
        return response

    memory_response = runtime._answer_memory_list_if_requested(user_input)
    if memory_response:
        runtime.conversation.add_assistant_message(memory_response)
        return memory_response

    # TOOL SHORTCUT
    tools_response = runtime._answer_tools_list_if_requested(user_input)
    if tools_response:
        runtime.conversation.add_assistant_message(tools_response)
        if runtime.memory_v2:
            runtime.memory_v2.store_episode(
                user_input, {"type": "tool_list", "output": tools_response}
            )
        return tools_response

    # MEMORY SHORTCUT
    memory_response = runtime.memory_manager.answer_from_memory(user_input)
    if memory_response:
        runtime.conversation.add_assistant_message(memory_response)
        if runtime.memory_v2:
            runtime.memory_v2.store_episode(
                user_input, {"type": "memory_hit", "output": memory_response}
            )
        return memory_response

    # PLAN INIT
    resume_plan = runtime._has_unfinished_plan() and runtime._is_plan_resume_request(
        user_input
    )
    if resume_plan and runtime.current_plan:
        for pending_task in runtime.current_plan.tasks:
            if pending_task.status == "in_progress" or (
                pending_task.status == "failed"
                and pending_task.attempts < pending_task.max_attempts
            ):
                pending_task.status = "pending"
        runtime._persist_plan_state()
    else:
        runtime.current_plan = await runtime.planner.create_plan(user_input)
        runtime._persist_plan_state()

    last_response = ""

    for _ in range(max_steps):
        # TASK PICK
        task = (
            runtime.scheduler.get_next_task(runtime.current_plan)
            if runtime.current_plan
            else None
        )
        task_query = task.description if task else user_input

        # MEMORY RETRIEVAL
        retrieved = runtime.memory_v2.retrieve(task_query) if runtime.memory_v2 else []
        memory_context = "\n".join(
            f"- {getattr(m, 'content', getattr(m, 'text', str(m)))}"
            for m in retrieved[:5]
        )

        # =========================
        # PROMPT BUILD
        # =========================
        system_prompt = runtime.prompter.build_system_prompt(
            runtime.tools,
            runtime.memory_manager.memory,
            user_input,
            plan=runtime.current_plan,
        )

        if memory_context:
            system_prompt += f"\n\nMEMORY V2:\n{memory_context}"

        # ✅ WORKING MEMORY
        working_context = runtime.working_memory.get_context_for_prompt()
        if working_context:
            system_prompt += working_context

        if task:
            system_prompt += f"\n\nTACHE ACTIVE:\n{task.description}"
            if task.status == "pending":
                task.status = "in_progress"
                runtime._persist_plan_state()

        # =========================
        # LLM CALL
        # =========================
        raw = await runtime._safe_chat(runtime._build_messages(system_prompt))
        action = runtime.parser.parse_action(raw)

        # =========================
        # ✅ REFUSER FINAL SI ACTION DEMANDÉE
        # =========================
        if action.tool == "final" and runtime._has_action_intent(user_input):
            response_text = str(action.args.get("content", ""))
            forced_tool, forced_args = runtime._get_forced_tool_for_intent(user_input)

            if forced_tool != "final" and runtime.tools.get(forced_tool):
                runtime.logger.warning(
                    f"Action demandée mais LLM répond final: "
                    f"'{response_text[:80]}...' → Forcer tool."
                )
                action.tool = forced_tool
                action.args = forced_args
                runtime.logger.info(
                    f"Tool forcé: {forced_tool} avec args {forced_args}"
                )

        # =========================
        # ANTI-ARGS-VIDE
        # =========================
        if action.tool and action.tool != "final":
            tool_ref = runtime.tools.get(action.tool)
            infer_args = getattr(tool_ref, "infer_args", None)
            if callable(infer_args):
                action.args = infer_args(user_input, action.args)

        if (
            action.tool
            and action.tool != "final"
            and runtime.tools.get(action.tool)
            and (not action.args or action.args == {})
        ):
            runtime.logger.warning(f"Args vides pour '{action.tool}'. Correction...")
            tool_info = ""
            tool_ref = runtime.tools.get(action.tool)
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
                raw = await runtime._safe_chat(
                    [
                        {"role": "system", "content": correction_prompt},
                    ]
                )
                action = runtime.parser.parse_action(raw)
            except Exception:
                pass

            if action.tool != "final" and (not action.args or action.args == {}):
                runtime.logger.error(
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
            if runtime._is_loop_detected(
                response_text, user_input
            ) and runtime._has_action_intent(user_input):
                forced_tool, forced_args = runtime._get_forced_tool_for_intent(
                    user_input
                )
                runtime.logger.warning(f"Boucle détectée → force '{forced_tool}'")
                action.tool = forced_tool
                action.args = forced_args

        if action.plan:
            for proposed_task in action.plan.tasks:
                proposed_task.status = "pending"
                proposed_task.result = None
                proposed_task.attempts = 0
            proposed_plan = runtime._merge_plan_update(action.plan)
            validated_plan = runtime.planner.validate_plan(proposed_plan)
            if validated_plan is None:
                runtime.logger.warning(
                    "Plan du LLM ignoré : dépendance inconnue, doublon, "
                    "cycle ou trop d'étapes."
                )
            else:
                runtime.current_plan = validated_plan
                runtime._persist_plan_state()

        tool_name = str(action.tool or "final").strip().casefold()
        args = action.args or {}

        # =========================
        # FINAL RESPONSE
        # =========================
        if tool_name == "final":
            response = str(args.get("content", ""))

            if runtime._needs_final_retry(response):
                response = await runtime._force_final_response(
                    system_prompt, "Réponds naturellement et brièvement."
                )

            if task:
                forced_tool, _ = runtime._get_forced_tool_for_intent(
                    f"{user_input} {task.description}"
                )
                if (
                    runtime._has_action_intent(f"{user_input} {task.description}")
                    and forced_tool != "final"
                    and runtime.tools.get(forced_tool)
                ):
                    task.status = "in_progress"
                    task.result = "Aucune réussite d'outil confirmée."
                    response = (
                        "Je ne peux pas confirmer que l'action a été exécutée; "
                        "je la laisse non terminée."
                    )
                    runtime._persist_plan_state()
                    runtime.conversation.add_assistant_message(response)
                    return response
                runtime._set_task_result(task, True, response)

            runtime.conversation.add_assistant_message(response)
            if runtime.memory_v2:
                runtime.memory_v2.store_episode(
                    user_input,
                    {
                        "type": "final",
                        "output": response,
                        "task": task.description if task else None,
                    },
                )

            if (
                task
                and runtime.current_plan
                and runtime.scheduler.get_next_task(runtime.current_plan)
            ):
                last_response = response
                continue

            return response

        # =========================
        # EMPTY TOOL FIX
        # =========================
        if tool_name in {"", "none", "null"}:
            response = await runtime._force_final_response(
                system_prompt, "Réponds directement à l'utilisateur."
            )
            if runtime.memory_v2:
                runtime.memory_v2.store_episode(
                    user_input,
                    {
                        "type": "direct",
                        "output": response,
                    },
                )
            runtime.conversation.add_assistant_message(response)
            return response

        # =========================
        # TOOL RESOLUTION
        # =========================
        tool = runtime.tools.get(tool_name)
        success = False
        observation = ""

        if not tool:
            runtime.logger.warning(
                f"Tool inconnu: '{tool_name}'. "
                f"Disponibles: {[t.name for t in runtime.tools.list_tools()]}"
            )
            available_tools = ", ".join(
                tool.name for tool in runtime.tools.list_tools()
            )
            observation = (
                f'Je n\'ai pas acces au tool "{tool_name}". '
                f"Tools disponibles: {available_tools}."
            )
            runtime.conversation.add_assistant_message(observation)
            return observation

        elif not runtime._is_tool_allowed(tool, user_input):
            observation = await runtime._force_final_response(
                system_prompt, f"Réponds à: {user_input}"
            )
            success = False

        else:
            try:
                start_time = time.time()
                observation = await runtime._run_tool(tool, args, user_input, trace_id)
                duration_ms = (time.time() - start_time) * 1000
                success = runtime._tool_result_succeeded(observation)

                # ✅ Enregistrer métrique
                runtime.evaluator.record_action(
                    metrics, tool_name, success, duration_ms, observation
                )
            except ToolAuthorizationError as e:
                response = str(e)
                metrics.status = "failed"
                runtime.evaluator.record_action(
                    metrics,
                    tool_name,
                    False,
                    0,
                    response,
                    error=response,
                )
                runtime.logger.warning(response)
                runtime.conversation.add_assistant_message(response)
                if runtime.event_bus:
                    await runtime.event_bus.emit(
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
                runtime.evaluator.record_action(
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
            evaluation = await runtime.reflector.evaluate(
                goal=goal,
                action_taken=f"{tool_name}({args})",
                result=observation,
                success=success,
            )

        if evaluation["status"] == "retry":
            runtime.evaluator.record_retry(metrics)
        elif evaluation["status"] == "replan":
            runtime.evaluator.record_replan(metrics)

        runtime.logger.info(
            f"[REFLECTION] status={evaluation['status']} | "
            f"analysis={evaluation['analysis'][:100]}"
        )

        # =========================
        # ✅ REPLANIFICATION (limité à 1 retry)
        # =========================
        retry_count = getattr(runtime, "_retry_count", 0)

        if evaluation["status"] in {"replan", "retry"} and runtime.current_plan:
            if retry_count >= 1:
                runtime.logger.warning(f"Max retries atteint ({retry_count}), abandon")
                response = (
                    f"Je n'ai pas réussi après {retry_count} tentative(s). "
                    f"Dernier résultat: {observation}"
                )
                runtime.conversation.add_assistant_message(response)
                if task:
                    runtime._set_task_result(task, False, observation)
                metrics.status = "failed"
                return response

            runtime._retry_count = retry_count + 1
            suggestion = evaluation.get("suggestion", "")
            runtime.logger.warning(
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
                runtime.current_plan = await runtime.planner.create_plan(
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
            runtime.conversation.add_assistant_message(response)
            if task:
                runtime._set_task_result(task, False, evaluation["analysis"])
            metrics.status = "failed"
            return response

        # TASK UPDATE
        if task:
            runtime._set_task_result(task, success, observation)

        # =========================
        # SKILL MEMORY
        # =========================
        if success and tool and runtime.memory_v2:
            runtime.memory_v2.store_skill(
                task=tool_name,
                steps=[task.description if task else tool_name, tool_name],
                success=True,
            )
            # ✅ WORKING MEMORY
            runtime.working_memory.put(f"resultat_{tool_name}", str(observation)[:200])
        # EPISODIC MEMORY
        if runtime.memory_v2:
            runtime.memory_v2.store_episode(
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
                and runtime.current_plan
                and runtime.scheduler.get_next_task(runtime.current_plan)
            ):
                runtime.conversation.add_assistant_message(observation)
                last_response = observation
                continue
            if not success:
                metrics.status = "failed"
            runtime.conversation.add_assistant_message(observation)
            return observation

        runtime.conversation.add_assistant_message(observation)
        last_response = observation

    metrics.status = "incomplete"
    response = f"Max steps atteints. Dernier état: {last_response}"
    runtime.conversation.add_assistant_message(response)
    return response
