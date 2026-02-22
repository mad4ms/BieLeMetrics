"""Compatibility entrypoint for Dagster Definitions.

The actively maintained Dagster setup lives in ``defs_debug.py``.
This module re-exports ``defs`` so tools that still resolve
``hbl_etl_dagster.defs`` continue to work.
"""

from .defs_debug import defs

__all__ = ["defs"]
