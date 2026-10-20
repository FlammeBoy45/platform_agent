# Agent Platform

Agent conversationnel en français, avec outils intégrés et mémoire persistante.

## Prérequis

- Python 3.12 ou plus
- [uv](https://docs.astral.sh/uv/)
- Une clé API Groq (développement), OpenAI (production), ou Ollama installé localement

## Installation et lancement

```bash
uv sync
```

Créez un fichier `.env` à la racine avec votre clé :

```env
GROQ_API_KEY=votre_cle_groq
SERPAPI_API_KEY=votre_cle_serpapi
```

Puis lancez l’agent :

```bash
uv run python -m src.main
```

Pour utiliser un modèle local avec Ollama, installe Ollama puis télécharge un modèle,
par exemple :

```bash
ollama pull llama3.2
```

Configure `.env` sans clé API cloud :

```env
APP_ENV=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
```

Lance ensuite Ollama (son service local écoute par défaut sur le port `11434`), puis
l’agent avec `uv run python -m src.main`. Le nom indiqué dans `OLLAMA_MODEL` doit être
disponible dans Ollama. En production cloud, définis `APP_ENV=production` et renseigne
`OPENAI_API_KEY`.
Les fichiers gérés par l’agent restent dans le dossier défini par `WORKSPACE_PATH`
(par défaut, la racine du projet), jamais dans un dossier temporaire. La mémoire est activée
par défaut (`ENABLE_MEMORY=false` pour la désactiver). L’exécution Python est désactivée
par défaut (`ENABLE_PYTHON_CODE=true` pour l’activer).
Le nom et les préférences détectés, ainsi que les souvenirs demandés explicitement,
sont enregistrés dans `data/memory.json`, partagé entre les modèles. Les faits expirent
après 365 jours et les épisodes de conversation après 30 jours. Demande « montre-moi ta
mémoire » pour consulter les souvenirs récents, « oublie [sujet] » pour en supprimer
un ou « oublie tout » pour vider la mémoire. Les anciennes entrées sans date sont
horodatées une seule fois lors de leur première lecture après cette mise à jour.
Cette mémoire correspond à un profil local : elle n'est pas isolée par utilisateur et
ne doit pas être partagée entre plusieurs utilisateurs avant l'ajout d'une authentification
et d'un identifiant utilisateur fiable. N'y enregistre pas de mots de passe, clés ou autres
secrets.
Le rôle local est `USER` par défaut (`AGENT_ROLE=readonly` ou `AGENT_ROLE=admin` pour
le modifier). Ce réglage n'est pas une authentification multi-utilisateur. Les créations,
modifications et suppressions de fichiers demandent une confirmation dans la console;
les lectures ne la demandent pas. Les appels du runtime sans gestionnaire de confirmation
sont refusés par défaut pour ces actions.
La commande `/version` affiche la version; `/new` démarre une session neuve.
La dernière conversation est rechargée au redémarrage depuis `data/sessions.jsonl`.
Les plans incomplets y sont sauvegardés; écris « continue » ou « reprends » pour relancer
le plan après redémarrage. `/new` démarre une session indépendante.
Des outils additionnels peuvent être fournis par des modules Python listés dans
`AGENT_PLUGIN_MODULES`; chaque module expose `register_tools(registry, settings, memory)`.
Chaque requête finalise ses métriques et émet `agent.completed`, `agent.failed` ou
`agent.interrupted` selon son issue; les réponses de raccourci sont aussi ajoutées
à l'historique de conversation.
Tu peux formuler les demandes naturellement, par exemple « fais-moi le module auth
même vide » : l’agent crée alors un squelette `.py` dans cet espace, sans écraser
un fichier existant.

## Tests

```bash
uv run pytest
```

## Contrôles qualité

Les mêmes contrôles sont exécutés automatiquement sur chaque push et pull request :

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -q
```
