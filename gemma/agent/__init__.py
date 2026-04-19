"""Agentic infrastructure for gemma-cli.

Houses the per-session tool-call cache and (in later phases) parallel
dispatch utilities. The public surface is intentionally small — callers
should import directly from the submodules they need.
"""
