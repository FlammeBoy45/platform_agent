# evaluator.py

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    """Résultat d'une action avec métriques."""
    tool_name: str
    success: bool
    duration_ms: float
    output_preview: str
    error: str | None = None


@dataclass
class SessionMetrics:
    """Métriques d'une session complète."""
    session_id: str
    start_time: float
    status: str = "running"
    end_time: float | None = None
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    retries: int = 0
    replans: int = 0
    total_duration_ms: float = 0.0
    actions: list[ActionResult] = field(default_factory=list)
    user_input: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_actions == 0:
            return 1.0
        return self.successful_actions / self.total_actions

    @property
    def avg_duration_ms(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.total_duration_ms / self.total_actions

    @property
    def session_duration_ms(self) -> float:
        end_time = self.end_time if self.end_time is not None else time.time()
        return (end_time - self.start_time) * 1000

    def summary(self) -> str:
        return (
            f"📊 Session {self.session_id[:8]}\n"
            f"   Statut: {self.status}\n"
            f"   Actions: {self.total_actions} "
            f"(✅ {self.successful_actions} | ❌ {self.failed_actions})\n"
            f"   Taux succès: {self.success_rate:.0%}\n"
            f"   Retries: {self.retries} | Replans: {self.replans}\n"
            f"   Durée totale: {self.total_duration_ms:.0f}ms "
            f"(moy: {self.avg_duration_ms:.0f}ms/action)"
        )


class ActionEvaluator:
    """
    Évalue et enregistre les métriques de performance des actions.
    Permet le suivi et l'amélioration continue.
    """

    def __init__(self, max_history: int = 100):
        self._history: list[SessionMetrics] = []
        self._max_history = max_history

    def start_session(self, session_id: str, user_input: str) -> SessionMetrics:
        """Démarre une nouvelle session de métriques."""
        metrics = SessionMetrics(
            session_id=session_id,
            start_time=time.time(),
            user_input=user_input,
        )
        return metrics

    def record_action(
        self,
        metrics: SessionMetrics,
        tool_name: str,
        success: bool,
        duration_ms: float,
        output: str,
        error: str | None = None,
    ) -> None:
        """Enregistre une action dans les métriques."""
        metrics.total_actions += 1
        if success:
            metrics.successful_actions += 1
        else:
            metrics.failed_actions += 1
        metrics.total_duration_ms += duration_ms

        metrics.actions.append(ActionResult(
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
            output_preview=output[:100],
            error=error,
        ))

    def record_retry(self, metrics: SessionMetrics) -> None:
        metrics.retries += 1

    def record_replan(self, metrics: SessionMetrics) -> None:
        metrics.replans += 1

    def end_session(self, metrics: SessionMetrics) -> str:
        """Termine la session et retourne un résumé."""
        if metrics.end_time is None:
            metrics.end_time = time.time()
        if not any(item.session_id == metrics.session_id for item in self._history):
            self._history.append(metrics)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        return metrics.summary()

    def get_global_stats(self) -> dict[str, Any]:
        """Retourne les statistiques globales sur toutes les sessions."""
        if not self._history:
            return {"sessions": 0}

        total_success = sum(s.successful_actions for s in self._history)
        total_actions = sum(s.total_actions for s in self._history)

        return {
            "sessions": len(self._history),
            "total_actions": total_actions,
            "total_success": total_success,
            "total_failed": total_actions - total_success,
            "global_success_rate": total_success / max(total_actions, 1),
            "avg_actions_per_session": total_actions / len(self._history),
        }