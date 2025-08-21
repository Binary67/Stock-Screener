# AGENTS Coding Contract

This document defines the non-negotiable coding rules for this repository. It exists to keep the codebase consistent, maintainable, and easy to extend.

## 1) PascalCase For All Newly Declared Variables

- Rule: Name every newly declared variable in PascalCase. This applies to local variables, function parameters, instance attributes, and module-level variables.
- Examples (Good → Bad):
  - `TotalCount` → not `total_count`
  - `DailyReturnsMap` → not `daily_returns_map`
  - `SourcePath` (parameter) → not `source_path`
  - `Self.PriceCache` → not `self.price_cache`
- Notes:
  - Do not rename existing variables just to conform—change only when you are already touching code for related reasons.
  - Third‑party APIs may force different names; keep external identifiers as required, but map them into PascalCase variables in our code.

## 2) DRY (Don’t Repeat Yourself)

- Consolidate duplication: If you copy code or config, extract it into a function, class, or shared module.
- Reuse helpers: Prefer calling existing utilities instead of re‑implementing similar logic.
- Centralize constants/config: Keep repeated literals (e.g., paths, URLs, column names) in a single source of truth.
- Single responsibility orchestration: Keep pipeline steps focused; compose them rather than duplicating logic across steps.
- Tests follow DRY too: Use fixtures/factories for repeated setup.

## 3) `main.py` Is The Orchestrator

- Single entrypoint: `main.py` is the script to run the whole pipeline.
- Always update `main.py` when adding a new module:
  - Import the new module(s) in `main.py`.
  - Register the new step in the pipeline run order.
  - Ensure any required configuration is wired through `main.py`.
- Keep `main.py` slim: It orchestrates; it should not contain business logic. Move logic into modules and call them from `main.py`.

To run the python script, always use ".venv/bin/python3" to run the py file.