import re

from src.domain.protocols.memory import LongTermMemory


class MemoryManager:
    """Gère l'extraction de faits et la récupération de données personnelles."""

    def __init__(self, memory: LongTermMemory | None) -> None:
        self.memory = memory
        self._patterns = (
            r"\bmon nom est\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '’-]{1,60})",
            r"\bje m'appelle\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '’-]{1,60})",
            r"\bappelle-moi\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '’-]{1,60})",
        )
        self._remember_pattern = re.compile(
            r"\b(?:souviens[- ]toi|retiens|mémorise|memorise|n'oublie pas)"
            r"(?:\s+que)?\s+(.+?)\s*[.!?]*$",
            flags=re.IGNORECASE,
        )
        self._save_memory_pattern = re.compile(
            r"\b(?:écris|ecris|mets|enregistre|garde|ajoute)"
            r"\s*(?:[- ]?(?:le|la|les|ça|ca|cela))?"
            r"\s+(?:dans|à|a)\s+(?:ta|tes|ma|mes)\s+m[eé]moires?\b",
            flags=re.IGNORECASE,
        )
        self._forget_all_pattern = re.compile(
            r"\b(?:oublie tout|efface toute ta mémoire|efface toute ma mémoire|"
            r"vide ta mémoire|supprime tous tes souvenirs|supprime toute ta mémoire)\b",
            flags=re.IGNORECASE,
        )
        self._forget_pattern = re.compile(
            r"\b(?:oublie|efface de ta mémoire|supprime de ta mémoire|"
            r"retire de ta mémoire)\s+(.+?)\s*[.!?]*$",
            flags=re.IGNORECASE,
        )
        self._preference_pattern = re.compile(
            r"\b(?:je préfère|je prefere|j'aime|j’aime|je n'aime pas|je n’aime pas|"
            r"je déteste|je deteste|ma préférence est)\s+(.+?)\s*[.!?]*$",
            flags=re.IGNORECASE,
        )
        self._agent_name_pattern = re.compile(
            r"\btoi\s+(?:c['’]est|cest)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*)",
            flags=re.IGNORECASE,
        )
        self._identity_statement_pattern = re.compile(
            r"\bje suis\s+(?P<user>.+?)\s+"
            r"(?:ton p[eè]re\s+)?et toi\s+"
            r"(?:c['’]est|cest|ces|tu es|t'es|es)\s+"
            r"(?P<agent>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*)\b",
            flags=re.IGNORECASE,
        )
        self._father_statement_pattern = re.compile(
            r"\bje suis\s+(?:toi\s+)?(?:ton p[eè]re|le p[eè]re)\b",
            flags=re.IGNORECASE,
        )
        self._father_of_agent_pattern = re.compile(
            r"\bje suis\s+(?:l{1,2}e\s+)?p[eè]re\s+de\s+"
            r"(?P<agent>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*)\b",
            flags=re.IGNORECASE,
        )
        self._agent_name_statement_pattern = re.compile(
            r"\btu\s+(?:es|est)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*)\b",
            flags=re.IGNORECASE,
        )
        self._agent_name_reverse_pattern = re.compile(
            r"\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*)\s+"
            r"(?:c['’]est|cest|ces)\s+toi\b",
            flags=re.IGNORECASE,
        )
        self._stored_fact_prefix = re.compile(
            r"^L'utilisateur a demandé de retenir\s*:\s*",
            flags=re.IGNORECASE,
        )
        self._restore_agent_identity()

    def learn_facts(self, user_input: str) -> str | None:
        if not self.memory:
            return None

        self._learn_identity(user_input)

        for pattern in self._patterns:
            match = re.search(pattern, user_input, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip(" .,!?:;")
                if name:
                    self.memory.remember(
                        key="user.name",
                        value=name,
                        text=f"Le nom de l'utilisateur est {name}.",
                    )
                    return None

        explicit_memory = self._remember_pattern.search(user_input)
        if explicit_memory:
            fact = explicit_memory.group(1).strip(" .,!?:;")
            if fact:
                self._remember_fact(fact)
                return fact

        preference = self._preference_pattern.search(user_input)
        if preference:
            detail = preference.group(1).strip(" .,!?:;")
            if detail:
                key = f"user.preference.{self._slug(detail)}"
                self.memory.remember(
                    key, detail, f"Préférence de l'utilisateur : {detail}."
                )
        return None

    def save_previous_statement(
        self, user_input: str, previous_user_input: str | None
    ) -> str | None:
        if (
            self.memory is None
            or previous_user_input is None
            or not self._save_memory_pattern.search(user_input)
        ):
            return None

        explicit_fact = self._remember_pattern.search(previous_user_input)
        fact = (
            explicit_fact.group(1).strip(" .,!?:;")
            if explicit_fact
            else previous_user_input.strip(" .,!?:;")
        )
        if not fact:
            return None

        self._remember_fact(fact)
        return fact

    def is_save_memory_request(self, user_input: str) -> bool:
        return bool(self._save_memory_pattern.search(user_input))

    def forget_requested_memory(self, user_input: str) -> tuple[bool, int] | None:
        if self._forget_all_pattern.search(user_input):
            return True, self.memory.forget() if self.memory else 0
        match = self._forget_pattern.search(user_input)
        if not match:
            return None
        query = match.group(1).strip(" .,!?:;")
        return True, self.memory.forget(query) if self.memory and query else 0

    def _remember_fact(self, fact: str) -> None:
        assert self.memory is not None
        key = f"user.fact.{self._slug(fact)}"
        self.memory.remember(key, fact, f"L'utilisateur a demandé de retenir : {fact}.")
        self._learn_identity(fact)

    def _restore_agent_identity(self) -> None:
        if self.memory is None:
            return
        for fact in self.memory.all():
            if fact.key.startswith("user.fact."):
                statement = self._stored_fact_prefix.sub("", fact.text).rstrip(". ")
                self._learn_identity(statement)

    def _learn_identity(self, statement: str) -> None:
        if self.memory is None:
            return

        father_of_agent = self._father_of_agent_pattern.search(statement)
        if father_of_agent:
            self._remember_agent_name(father_of_agent.group("agent"))
            self._remember_agent_father()
            return

        identity = self._identity_statement_pattern.search(statement)
        if identity:
            user_name = identity.group("user").strip(" .,!?:;")
            agent_name = identity.group("agent")
            if user_name:
                self.memory.remember(
                    "user.name",
                    user_name,
                    f"Le nom de l'utilisateur est {user_name}.",
                )
            self._remember_agent_name(agent_name)
            if re.search(r"\bton p[eè]re\b", statement, flags=re.IGNORECASE):
                self._remember_agent_father(user_name)
            return

        if self._father_statement_pattern.search(statement):
            self._remember_agent_father()

        agent_name = self._agent_name_pattern.search(statement)
        if agent_name:
            self._remember_agent_name(agent_name.group(1))
            return

        agent_name = self._agent_name_statement_pattern.search(statement)
        if agent_name:
            self._remember_agent_name(agent_name.group(1))
            return

        agent_name = self._agent_name_reverse_pattern.search(statement)
        if agent_name:
            self._remember_agent_name(agent_name.group(1))

    def _remember_agent_name(self, name: str) -> None:
        assert self.memory is not None
        normalized_name = name.strip(" .,!?:;")
        if normalized_name:
            self.memory.remember(
                "agent.name",
                normalized_name,
                f"Le nom choisi pour l'assistant est {normalized_name}.",
            )

    def _remember_agent_father(self, name: str | None = None) -> None:
        assert self.memory is not None
        user_name = self.memory.get("user.name")
        father = name or (user_name.value if user_name else "toi, l'utilisateur")
        self.memory.remember(
            "agent.father",
            father,
            f"L'utilisateur indique être le père de l'assistant : {father}.",
        )

    @staticmethod
    def _slug(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        return normalized[:80] or "souvenir"

    def _answer_identity_relation(self, normalized: str) -> str | None:
        if any(
            phrase in normalized
            for phrase in (
                "pas la peine d'en rajouter",
                "pas la peine den rajouter",
                "pas besoin d'en rajouter",
                "pas besoin den rajouter",
            )
        ):
            return "Compris, je n'en rajoute pas."

        agent_name = self.memory.get("agent.name") if self.memory else None
        named_person = re.search(
            r"\bqui est (?:l['’])?([\wÀ-ÿ'-]+)\b",
            normalized,
        )
        if (
            agent_name
            and named_person
            and named_person.group(1).strip("'’") == agent_name.value.casefold()
        ):
            return f"{agent_name.value.capitalize()}, c'est moi, ton assistant IA."

        asks_father = re.search(
            r"\b(?:et\s+)?(?:qui|sui)\s+est\s+"
            r"(?:(?:le|son|ton)\s+)?p[eè]re\b",
            normalized,
        )
        if asks_father:
            father = self.memory.get("agent.father") if self.memory else None
            if father:
                if father.value == "toi, l'utilisateur":
                    return "Tu m'as dit que tu es mon père."
                return f"Tu m'as dit que {father.value} est mon père."
            return "Je n'ai pas mémorisé qui est mon père."

        return None

    def answer_from_memory(self, user_input: str) -> str | None:
        if not self.memory:
            return None

        normalized = user_input.casefold()
        relationship_answer = self._answer_identity_relation(normalized)
        if relationship_answer:
            return relationship_answer

        asks_agent_identity = any(
            phrase in normalized
            for phrase in (
                "qui es-tu",
                "qui es tu",
                "qui est tu",
                "qui est-tu",
                "comment tu t'appelles",
                "comment t'appelles-tu",
                "comment t'appelles tu",
                "ton nom",
                "tu t'appelles comment",
                "tu t'appelle comment",
            )
        )
        if asks_agent_identity:
            agent_name = self.memory.get("agent.name")
            if agent_name:
                return f"Je suis {agent_name.value}, ton assistant IA."
            return "Je suis ton assistant IA."

        asks_name = any(
            phrase in normalized
            for phrase in (
                "mon nom",
                "qui suis-je",
                "qui suis je",
                "qui je suis",
                "je m'appelle comment",
                "je m'appelle qui",
            )
        )

        if not asks_name:
            return None

        name = self.memory.get("user.name")
        if not name:
            return "Je ne connais pas encore ton nom."

        return f"Ton nom est {name.value}."

    def recall_context(self, user_input: str, limit: int = 8) -> str:
        if not self.memory:
            return ""
        facts = self.memory.search(user_input, limit=limit)
        if not facts:
            return ""
        return "\n".join(f"- {fact.text}" for fact in facts)
