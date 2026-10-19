from src.domain.entities.plan import Plan, Task
from src.domain.protocols.llm_provider import LLMProvider
from src.application.orchestrators.response_parser import ResponseParser
from src.domain.protocols.memory import AgentMemory


class Planner:
    def __init__(
        self,
        llm: LLMProvider,
        parser: ResponseParser,
        memory_v2: AgentMemory | None = None,
    ):
        self.llm = llm
        self.parser = parser
        self.memory_v2 = memory_v2

    async def create_plan(self, goal: str) -> Plan:
        skills_context = ""

        if self.memory_v2:
            try:
                skills = self.memory_v2.retrieve_skills(goal)
            except Exception:
                skills = []

            if skills:
                skills_context = "\n".join(
                    f"- Tâche: {getattr(s, 'content', str(s))}"
                    for s in skills[:5]
                )

        prompt = f"""
Tu es un planificateur. Analyse l'objectif et découpe-le en étapes simples.

OBJECTIF: {goal}

COMPÉTENCES APPRISES:
{skills_context or "Aucune"}

Retourne UNIQUEMENT ce JSON:
{{
    "goal": "{goal}",
    "tasks": [
        {{"description": "étape 1", "depends_on": [], "status": "pending"}},
        {{"description": "étape 2", "depends_on": [], "status": "pending"}}
    ]
}}

RÈGLES:
- Si l'objectif est simple (salut, question), crée UNE SEULE tâche
- depends_on doit être une liste de strings (descriptions de tâches), JAMAIS des indices numériques
- status toujours "pending"
"""

        try:
            raw = await self.llm.chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": goal},
            ])
        except Exception as e:
            # Fallback si le LLM échoue
            return Plan(
                goal=goal,
                tasks=[Task(description=goal, depends_on=[], status="pending")]
            )

        parsed = self.parser._extract_json(raw)

        if not isinstance(parsed, dict):
            return Plan(
                goal=goal,
                tasks=[Task(description=goal, depends_on=[], status="pending")]
            )

        tasks: list[Task] = []
        raw_tasks = parsed.get("tasks", [])

        # ✅ Si le plan est un dict numérique {"1": "...", "2": "..."}
        if not raw_tasks or not isinstance(raw_tasks, list):
            numeric_keys = sorted(
                [k for k in parsed.keys() if k.isdigit()],
                key=int
            )
            if numeric_keys:
                raw_tasks = [
                    {"description": str(parsed[k]), "depends_on": [], "status": "pending"}
                    for k in numeric_keys
                ]

        for t in raw_tasks:
            if not isinstance(t, dict):
                continue

            desc = str(t.get("description", "")).strip()
            if not desc:
                continue

            depends = t.get("depends_on", [])
            # ✅ Nettoyage : ne garder que les strings, ignorer les int/None
            if isinstance(depends, list):
                depends = [
                    str(d) for d in depends
                    if d is not None and not isinstance(d, (int, float))
                ]
            else:
                depends = []

            tasks.append(
                Task(
                    description=desc,
                    depends_on=depends,
                    status="pending",
                )
            )

        # ✅ Si aucune tâche valide, créer une tâche par défaut
        if not tasks:
            tasks.append(
                Task(
                    description=goal,
                    depends_on=[],
                    status="pending",
                )
            )

        plan_goal = str(parsed.get("goal", goal)).strip() or goal

        return Plan(goal=plan_goal, tasks=tasks)
