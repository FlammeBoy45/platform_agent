from src.domain.entities.plan import Plan, Task


class Scheduler:
    def get_next_task(self, plan: Plan) -> Task | None:
        completed = {
            identifier
            for task in plan.tasks
            if task.status == "completed"
            for identifier in (task.id, task.description)
        }

        for task in plan.tasks:
            if task.status != "pending":
                continue

            if all(dep in completed for dep in task.depends_on):
                return task

        return None
