import json
from typing import Any

from src.domain.entities.tool_call import ToolCall
from src.domain.protocols.logger import LoggerProtocol


class ResponseParser:
    """Responsable de l'extraction et de la validation des réponses du LLM."""

    def __init__(self, logger: LoggerProtocol) -> None:
        self.logger = logger

    # ----------------------------
    # MAIN ENTRY
    # ----------------------------
    def parse_action(self, raw: str) -> ToolCall:
        parsed = self._extract_json(raw)

        # 🧨 FIX 1 : fallback strict immédiat
        if not isinstance(parsed, dict):
            return ToolCall(
                tool="final",
                args={"content": raw},
                plan=None,
            )

        # 🧠 log reasoning
        thought = parsed.pop("thought", None) or parsed.pop("reasoning", None)
        if thought:
            self.logger.info(f"[AGENT-REASONING] {thought}")

        try:
            parsed = self._normalize_response_shape(parsed)
            parsed = self._normalize_args(parsed)
            parsed = self._force_tool_field(parsed)  # 🔥 FIX IMPORTANT
            parsed = self._extract_plan_from_args(parsed)
            parsed = self._sanitize_plan(parsed)

            return ToolCall.model_validate(parsed)

        except Exception as e:
            self.logger.warning(f"Validation Pydantic échouée: {e}")

        return ToolCall(
            tool="final",
            args={"content": raw.strip()},
            plan=None,
        )

    @staticmethod
    def _normalize_response_shape(parsed: dict[str, Any]) -> dict[str, Any]:
        """Accept common final-response shapes returned by compatible models."""
        if isinstance(parsed.get("tool"), str):
            return parsed

        response_type = parsed.get("response")
        if response_type == "final" and isinstance(parsed.get("text"), str):
            parsed["tool"] = "final"
            parsed["args"] = {"content": parsed["text"]}
            return parsed

        for field in ("message", "content"):
            value = parsed.get(field)
            if isinstance(value, str):
                parsed["tool"] = "final"
                parsed["args"] = {"content": value}
                return parsed

        return parsed

    # ----------------------------
    # FIX 1: args normalization
    # ----------------------------
    def _normalize_args(self, parsed: dict[str, Any]) -> dict[str, Any]:
        args = parsed.get("args", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        if args is None or not isinstance(args, dict):
            args = {}

        parsed["args"] = args
        return parsed

    # ----------------------------
    # 🔥 FIX 2: FORCE TOOL FIELD (TON BUG ACTUEL)
    # ----------------------------
    def _force_tool_field(self, parsed: dict[str, Any]) -> dict[str, Any]:
        tool = parsed.get("tool")

        if not tool or not isinstance(tool, str):
            parsed["tool"] = "final"

        return parsed

    # ----------------------------
    # FIX 3: extract plan from args
    # ----------------------------
    def _extract_plan_from_args(self, parsed: dict[str, Any]) -> dict[str, Any]:
        args = parsed.get("args")

        if isinstance(args, dict) and "plan" in args:
            parsed["plan"] = args.pop("plan")

        return parsed

    # ----------------------------
    # FIX 4: sanitize plan
    # ----------------------------
    def _sanitize_plan(self, parsed: dict[str, Any]) -> dict[str, Any]:
        plan = parsed.get("plan")

        # ✅ CAS 0 : plan est une string → le convertir en dict
        if isinstance(plan, str):
            plan = {"goal": plan, "tasks": [{"description": plan, "status": "pending"}]}
            parsed["plan"] = plan

        if not isinstance(plan, dict):
            return parsed

        plan.pop("args", None)

        tasks = plan.get("tasks")

        if isinstance(tasks, list):
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                status = str(t.get("status", "")).lower()
                if status in {"done", "finished", "success"}:
                    t["status"] = "completed"
                elif status in {"todo", "pending"}:
                    t["status"] = "pending"
                elif status in {"doing", "in_progress"}:
                    t["status"] = "in_progress"
                elif status in {"error", "failed"}:
                    t["status"] = "failed"
            return parsed

        numeric_keys = sorted([k for k in plan.keys() if k.isdigit()], key=int)
        if numeric_keys:
            plan["tasks"] = [
                {"description": str(plan[k]), "status": "pending"} for k in numeric_keys
            ]
            if "goal" not in plan:
                plan["goal"] = plan["tasks"][0]["description"]
            for k in numeric_keys:
                plan.pop(k, None)
            return parsed

        if not tasks:
            plan["tasks"] = [{"description": str(plan), "status": "pending"}]
            if "goal" not in plan:
                plan["goal"] = "Tâche en cours"

        return parsed  # ----------------------------

    # JSON extraction (robust)
    # ----------------------------
    def _extract_json(self, raw: str) -> Any:
        text = raw.strip()
        decoder = json.JSONDecoder()

        # 1 direct parse
        try:
            value, _ = decoder.raw_decode(text)
            return value
        except Exception:
            pass

        # 2 markdown block
        if text.startswith("```"):
            cleaned = text.strip("`").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            try:
                value, _ = decoder.raw_decode(cleaned)
                return value
            except Exception:
                pass

        # 3 brute scan
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    value, _ = decoder.raw_decode(text[i:])
                    if isinstance(value, dict):
                        return value
                except Exception:
                    continue

        # 🔥 FIX FINAL: never return {} silently (causes hidden bugs)
        return None
