# Repository Guidelines

## Project Structure & Module Organization

Backend code lives in `src/sandevistan_read/`. Keep HTTP routes in `app.py`, document parsing in `documents.py`, retrieval logic in `retrieval.py`, background work in `jobs.py`, and provider integrations in `providers.py`. The React 19/Vite UI is under `frontend/src/`; shared overlays and rendering helpers belong in `ui.tsx`, while route-level management views belong in `management.tsx`. Python tests live in `tests/`, including the Playwright smoke test. Treat `runtime/`, `.venv/`, `.tools/`, `frontend/dist/`, and `frontend/node_modules/` as generated or local-state directories; do not commit their contents or use production data as fixtures.

## Build, Test, and Development Commands

- `./scripts/bootstrap.sh` installs the project environment, frontend dependencies, and local media tools.
- `./scripts/start.sh` starts the built application; `./scripts/stop.sh` stops it.
- `.venv/bin/pytest -q` runs the Python test suite.
- `corepack pnpm --dir frontend dev` runs the Vite development server.
- `corepack pnpm --dir frontend lint` performs TypeScript checking.
- `corepack pnpm --dir frontend build` creates the production UI in `frontend/dist/`.
- `.venv/bin/python tests/browser_smoke.py` runs read-only Chromium UI checks against a running server on port `20830`.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for Python functions and modules; use `PascalCase` for React components and `camelCase` for TypeScript values. Add type annotations to public Python APIs and explicit TypeScript prop types. Keep API calls in `frontend/src/api.ts`, avoid duplicating overlay behavior, and preserve the existing local-first, source-grounded boundaries. TypeScript checking is the frontend lint gate; match surrounding formatting when editing compact existing components.

## Testing Guidelines

Use pytest files named `test_*.py` and test functions named `test_*`. Add focused tests for parsing, retrieval, job state, or API behavior whenever those areas change. UI changes should extend `tests/browser_smoke.py` with desktop and mobile assertions, including keyboard behavior and horizontal-overflow checks. Do not make smoke tests delete notebooks, upload files, or trigger paid/external providers.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout. Use short, imperative commit subjects such as `Fix nested dialog focus handling`, and keep unrelated changes separate. Pull requests should explain user-visible behavior, list verification commands, link relevant issues, and include before/after screenshots for UI changes. Call out schema, configuration, provider, or runtime-data impacts explicitly.

## Security & Configuration

Never commit `runtime/config.toml`, access keys, provider credentials, uploaded documents, databases, or generated media. Update `config.example.toml` when adding configuration. Non-localhost deployments must configure `security.access_key`; document when a provider sends document context outside the local machine.
