# Repository Guidelines

## Project Structure & Module Organization
Code that models CrossRing fabrics lives in `src/core`, while traffic preparation pipelines are in `src/traffic_process` and shared helpers reside under `src/utils`. Configuration presets are stored in `config/`, ready-made reference experiments in `example/`, and automation or analysis helpers in `scripts/`. Simulation artifacts land in `Result/` and supporting datasets are under `traffic/` and `test_data/`. Pytest suites live in `test/` and should mirror the layout of the modules they cover.

## Build, Test, and Development Commands
Create a virtual environment of your choice, install dependencies, then run the package in editable mode with `pip install -e .`. Use `python example/example.py` to execute the default reference scenario; adjust paths inside the script for custom traffic. Run the regression suite with `pytest`, or target a module via `pytest test/test_arbitration.py`. For quick linting before review, run `python -m compileall src` to catch syntax regressions.

## Coding Style & Naming Conventions
Match existing Python style: PEP 8 spacing, 4-space indents, and docstrings on public classes or functions. Favor descriptive snake_case for functions and variables, CamelCase for classes, and keep module names lowercase. Preserve current folder-level `__init__.py` exports so new models surface through the same import paths. When touching configuration files, keep JSON and YAML keys lowercase with underscores, and align comments in bilingual (English/Chinese) style where present.

## Testing Guidelines
Extend tests in `test/` using pytest. Name files `test_<feature>.py` and fixtures in snake_case. Cover both routing logic and traffic preprocessing branches when adding a feature, and assert on counters stored in Result outputs if behavior changes. Run `pytest -k <keyword>` locally before pushing, and add sample traffic to `test_data/` only if it is under 100 KB and anonymized.

## Commit & Pull Request Guidelines
Recent commits are single-sentence summaries (often bilingual) such as `Bug: 修复D2D tracker释放`. Follow that format: prefix with a concise category when helpful (`Bug:`, `Feat:`, `Docs:`), keep the subject under 60 characters, and prefer Mandarin descriptions when touching CN-facing docs. Each pull request should link to related issues, describe simulation scenarios exercised, call out configuration or traffic files added, and include plots or table snapshots when result metrics change.

## Configuration & Reproducibility Tips
Store reusable parameter sets in `config/` and reference them by relative path inside scripts; avoid embedding absolute workstation paths. When a model exposes a `seed` field (for example in `src/utils/routing_strategies.py`), set it explicitly in experiments so regression runs stay deterministic. When sharing large result archives, upload elsewhere and place download instructions in `docs/` instead of adding binaries to the repo.
