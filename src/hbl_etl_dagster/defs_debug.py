"""Compatibility entrypoint.

Primary Dagster definitions now live in ``defs.py``.
"""

from .defs import defs

__all__ = ["defs"]
