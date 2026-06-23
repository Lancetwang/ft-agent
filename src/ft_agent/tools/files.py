"""Compatibility wrapper for agent_core.tools.files."""

from agent_core.tools.files import (
    build_edit_file_tool,
    build_file_tools,
    build_read_file_tool,
    build_write_file_tool,
)

__all__ = [
    "build_edit_file_tool",
    "build_file_tools",
    "build_read_file_tool",
    "build_write_file_tool",
]
