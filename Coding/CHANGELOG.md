# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2026-02-25

### Added
- `start.sh` — One-command startup script (creates venv, installs deps, launches app)
- `streamlit-autorefresh` to `requirements.txt` (was imported but missing)
- `.gitignore` with Python/Streamlit-specific entries
- Virtual environment setup instructions in `README.md`

### Changed
- `README.md` — Quickstart now shows `./start.sh` plus manual venv steps
- `Dockerfile` — Updated base image from Python 3.9 to 3.13, added `curl` for healthcheck
