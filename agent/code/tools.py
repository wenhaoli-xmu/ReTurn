def agent_instructions(root="/testbed"):
    return f"""You are a coding agent working in a sandboxed repository at {root}.
Resolve the user's issue by inspecting the repository, editing the implementation, and running
relevant tests. Do not only explain a solution: make the changes in the repository. Use dedicated
file tools instead of shell commands when possible. Use run_shell_command for tests, git, and other
terminal operations. Paths passed to file tools should be absolute and remain under {root}.
Do not access the network, install the target repository from elsewhere, commit changes, or modify
git history. Preserve unrelated work. When the task is complete, briefly summarize the changes and
tests; the harness will collect the patch from git diff. Work narrowly on the requested issue and
keep reasoning and tool calls concise. Once the relevant tests pass, stop using tools and return
the final summary; do not continue exploring related but unrequested code."""


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool_definitions(root="/testbed"):
    return [
        tool("list_directory", "List files and directories directly under a repository path.", {
            "path": {"type": "string", "description": f"Absolute directory path under {root}."},
            "ignore": {"type": "array", "items": {"type": "string"}},
            "respect_git_ignore": {"type": "boolean", "default": True},
        }, ["path"]),
        tool("read_file", "Read a text file, optionally selecting a line range.", {
            "file_path": {"type": "string", "description": f"Absolute file path under {root}."},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1},
        }, ["file_path"]),
        tool("write_file", "Create or overwrite a text file.", {
            "file_path": {"type": "string", "description": f"Absolute file path under {root}."},
            "content": {"type": "string"},
        }, ["file_path", "content"]),
        tool("edit", "Replace exact literal text in a file.", {
            "file_path": {"type": "string", "description": f"Absolute file path under {root}."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "default": False},
        }, ["file_path", "old_string", "new_string"]),
        tool("glob", "Find repository files matching a glob pattern.", {
            "pattern": {"type": "string", "description": "Glob such as src/**/*.py."},
            "path": {"type": "string", "description": f"Search root under {root}."},
        }, ["pattern"]),
        tool("grep_search", "Search repository text with a case-insensitive regular expression.", {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": f"File or directory under {root}."},
            "glob": {"type": "string", "description": "Optional file glob."},
            "limit": {"type": "integer", "minimum": 1},
        }, ["pattern"]),
        tool("run_shell_command", f"Run a one-shot bash command in {root} and return output.", {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 1200},
            "is_background": {"type": "boolean", "default": False},
        }, ["command"]),
    ]


AGENT_INSTRUCTIONS = agent_instructions()
TOOLS = tool_definitions()


_ARGS = {
    "list_directory": ({"path"}, {"path", "ignore", "respect_git_ignore"}),
    "read_file": ({"file_path"}, {"file_path", "offset", "limit"}),
    "write_file": ({"file_path", "content"}, {"file_path", "content"}),
    "edit": ({"file_path", "old_string", "new_string"},
             {"file_path", "old_string", "new_string", "replace_all"}),
    "glob": ({"pattern"}, {"pattern", "path"}),
    "grep_search": ({"pattern"}, {"pattern", "path", "glob", "limit"}),
    "run_shell_command": ({"command"},
                          {"command", "description", "timeout", "is_background"}),
}


async def execute(sandbox, name, params):
    if name not in _ARGS or not isinstance(params, dict):
        raise ValueError(f"unknown tool: {name}")
    required, allowed = _ARGS[name]
    missing, extra = required - set(params), set(params) - allowed
    if missing or extra:
        raise ValueError(f"invalid {name} arguments: missing={sorted(missing)} extra={sorted(extra)}")
    return await getattr(sandbox, name)(**params)
