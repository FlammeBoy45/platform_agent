# Agent Platform

Agent conversationnel en français, avec outils intégrés et mémoire persistante.

## Prérequis

- Python 3.12 ou plus
- [uv](https://docs.astral.sh/uv/)
- Une clé API Groq (développement) ou OpenAI (production)

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

En production, définissez `APP_ENV=production` et renseignez `OPENAI_API_KEY`.
Les fichiers gérés par l’agent restent dans le dossier défini par `WORKSPACE_PATH`
(par défaut, la racine du projet), jamais dans un dossier temporaire. La mémoire est activée
par défaut (`ENABLE_MEMORY=false` pour la désactiver). L’exécution Python est désactivée
par défaut (`ENABLE_PYTHON_CODE=true` pour l’activer).
Les échanges et souvenirs explicites sont enregistrés au fil des conversations dans
`data/memory.json`, partagé entre les modèles. Demande « montre-moi ta mémoire » pour
afficher les souvenirs récents.
Le rôle local est `USER` par défaut (`AGENT_ROLE=readonly` ou `AGENT_ROLE=admin` pour
le modifier). Ce réglage n'est pas une authentification multi-utilisateur. Les créations,
modifications et suppressions de fichiers demandent une confirmation dans la console;
les lectures ne la demandent pas. Les appels du runtime sans gestionnaire de confirmation
sont refusés par défaut pour ces actions.
Tu peux formuler les demandes naturellement, par exemple « fais-moi le module auth
même vide » : l’agent crée alors un squelette `.py` dans cet espace, sans écraser
un fichier existant.

## Tests

```bash
uv run pytest
```
