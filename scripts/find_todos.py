#!/usr/bin/env python3
"""Script to find and document all TODO comments in the codebase."""

import os
import re
from pathlib import Path
from datetime import datetime


def find_todos(directory: Path) -> list[tuple[str, int, str]]:
    """Find all TODO comments in the codebase.

    Args:
        directory: Root directory to search

    Returns:
        List of tuples containing (file_path, line_number, todo_text)
    """
    todos = []
    todo_pattern = re.compile(r"#\s*TODO:?\s*(.+)$")

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    match = todo_pattern.search(line)
                    if match:
                        # Get the relative path from the project root
                        rel_path = file_path.relative_to(directory)
                        todos.append((str(rel_path), i, match.group(1).strip()))

    return todos


def generate_todo_markdown(todos: list[tuple[str, int, str]], output_file: Path):
    """Generate a markdown file documenting all TODOs.

    Args:
        todos: List of (file_path, line_number, todo_text) tuples
        output_file: Path to write the markdown file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Project TODOs\n\n")
        f.write(f'*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*\n\n')

        if not todos:
            f.write("No TODOs found in the codebase.\n")
            return

        # Group TODOs by file
        todos_by_file = {}
        for file_path, line_num, todo_text in todos:
            if file_path not in todos_by_file:
                todos_by_file[file_path] = []
            todos_by_file[file_path].append((line_num, todo_text))

        # Write TODOs grouped by file
        for file_path, file_todos in sorted(todos_by_file.items()):
            f.write(f"## {file_path}\n\n")
            for line_num, todo_text in sorted(file_todos):
                f.write(f"- Line {line_num}: {todo_text}\n")
            f.write("\n")


def main():
    """Main entry point."""
    # Get the project root (parent of the scripts directory)
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    # Find all TODOs
    todos = find_todos(src_dir)

    # Generate the markdown file
    output_file = project_root / "TODO.md"
    generate_todo_markdown(todos, output_file)

    print(f"Found {len(todos)} TODOs")
    print(f"Generated TODO.md at {output_file}")


if __name__ == "__main__":
    main()
