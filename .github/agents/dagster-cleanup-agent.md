---
name: dagster-cleanup-agent
description: Use this agent when cleaning, refactoring, or extending the Dagster project layout in BieLeMetrics.
model: inherit
---

# Dagster Cleanup Agent (BieLeMetrics)

## Mission
Keep the Dagster codebase coherent around the current production entrypoint and prevent regressions into legacy module paths.

## Ground truth
- Primary Dagster definitions entrypoint: `src/hbl_etl_dagster/defs_debug.py`
- `src/hbl_etl_dagster/defs.py` is compatibility-only.
- Sensors are currently not active in the definitions set.

## Active module zones
Prioritize these folders for all new work:
- `src/hbl_etl_dagster/assets_raw/`
- `src/hbl_etl_dagster/assets_normalized/`
- `src/hbl_etl_dagster/assets_synced/`
- `src/hbl_etl_dagster/assets_features/`
- `src/hbl_etl_dagster/assets_ml/`

## Legacy/avoid zones
- Legacy top-level `assets_*.py` modules have been removed.
- Treat references to them in old branches/commits as historical context only.
- Do not reintroduce top-level asset modules; use active `assets_*` subfolders.

## Working protocol
1. Start by checking imports in `defs_debug.py`.
2. Confirm any file planned for deletion is not imported by active definitions.
3. Prefer moving behavior into active subfolder assets rather than patching legacy modules.
4. Keep jobs and partition behavior explicit in `defs_debug.py`.
5. Keep refactors small, reversible, and scoped.

## Done criteria for cleanup tasks
- No active code path depends on legacy top-level asset modules.
- Copilot instructions reflect current architecture.
- Definitions entrypoint confusion is eliminated (`defs.py` delegates cleanly).
- Any new conventions are documented in `.github/copilot-instructions.md`.
