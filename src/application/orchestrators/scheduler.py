from src.domain.entities.plan import Plan


class Scheduler:
    def get_next_task(self, plan: Plan):
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
