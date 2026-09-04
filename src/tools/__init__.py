"""Importing this package registers the tool catalog.

Without this, `registry` is empty for any caller that imported the graph but
not the catalog, and the planner silently produces an empty plan.
"""

from . import catalog
from .registry import Tool, ToolRegistry, registry, tool

__all__ = ["Tool", "ToolRegistry", "catalog", "registry", "tool"]
