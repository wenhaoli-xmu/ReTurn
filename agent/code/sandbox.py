import asyncio
import json
import re
import uuid


_HELPER = r'''
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else "/testbed").resolve()
op = sys.argv[1]
arg = json.load(sys.stdin)


def path(raw="."):
    p = Path(raw)
    p = (p if p.is_absolute() else ROOT / p).resolve(strict=False)
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"path escapes {ROOT}: {raw}")
    return p


def boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


if op == "read_file":
    p = path(arg["file_path"])
    data = p.read_bytes()
    if b"\0" in data:
        raise ValueError(f"cannot display binary file: {p}")
    lines = data.decode("utf-8", "replace").splitlines(keepends=True)
    offset = int(arg.get("offset", 0))
    limit = int(arg.get("limit", 2000))
    print(f"[lines {offset + 1}-{min(offset + limit, len(lines))} of {len(lines)}]")
    print("".join(lines[offset:offset + limit]), end="")

elif op == "write_file":
    p = path(arg["file_path"])
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(arg["content"]), encoding="utf-8")
    print(f"Successfully {'overwrote' if existed else 'created'} file: {p}")

elif op == "edit":
    p = path(arg["file_path"])
    old, new = str(arg["old_string"]), str(arg["new_string"])
    if not old:
        if p.exists():
            raise ValueError(f"file already exists: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new, encoding="utf-8")
        print(f"Created new file: {p}")
    else:
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        replace_all = boolean(arg.get("replace_all", False))
        if count == 0:
            raise ValueError("old_string was not found")
        if count > 1 and not replace_all:
            raise ValueError(f"old_string has {count} matches; add context or set replace_all")
        p.write_text(text.replace(old, new, -1 if replace_all else 1), encoding="utf-8")
        print(f"Successfully modified file: {p} ({count if replace_all else 1} replacements)")

elif op == "glob":
    base = path(arg.get("path", "."))
    pattern = str(arg["pattern"])
    files = [p for p in base.glob(pattern) if p.is_file() and ".git" not in p.parts]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print("\n".join(str(p) for p in files[:100]) or "No files found")
    if len(files) > 100:
        print(f"[{len(files) - 100} files truncated]")

elif op == "list_directory":
    base = path(arg["path"])
    ignore = arg.get("ignore", [])
    if isinstance(ignore, str):
        try:
            ignore = json.loads(ignore)
        except json.JSONDecodeError:
            ignore = [ignore]
    rows = []
    for p in base.iterdir():
        if p.name == ".git" or any(fnmatch.fnmatch(p.name, pat) for pat in ignore):
            continue
        rows.append((not p.is_dir(), p.name, f"[DIR] {p.name}" if p.is_dir() else p.name))
    print(f"Directory listing for {base}:\n" + "\n".join(x[2] for x in sorted(rows)))

elif op == "grep_search":
    target = path(arg.get("path", "."))
    pattern = re.compile(str(arg["pattern"]), re.I)
    file_glob = arg.get("glob")
    limit = int(arg.get("limit", 100))
    files = [target] if target.is_file() else (
        Path(root) / name
        for root, dirs, names in os.walk(target)
        if not (dirs.__setitem__(slice(None), [d for d in dirs if d != ".git"]) or False)
        for name in names)
    rows = []
    for p in files:
        rel = str(p.relative_to(ROOT))
        if file_glob and not (fnmatch.fnmatch(rel, file_glob) or fnmatch.fnmatch(p.name, file_glob)):
            continue
        try:
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    rows.append(f"{rel}:{n}:{line[:2000]}")
                    if len(rows) >= limit:
                        break
        except (OSError, UnicodeError):
            pass
        if len(rows) >= limit:
            break
    print("\n".join(rows) or "No matches found")
'''


def image_name(instance_id, namespace="swebench", arch="x86_64"):
    name = instance_id.lower().replace("__", "_1776_")
    return f"{namespace}/sweb.eval.{arch}.{name}:latest"


class Sandbox:
    def __init__(self, image, cpus=4, memory="16g", pids=512,
                 command_timeout=300, max_output=60000, network="none",
                 docker_args=None, harden=True):
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self.command_timeout = command_timeout
        self.max_output = max_output
        self.network = network
        self.docker_args = list(docker_args or [])
        self.harden = harden
        safe = re.sub(r"[^a-z0-9_.-]", "-", image.lower().rsplit("/", 1)[-1])[:48]
        self.name = f"code-{safe}-{uuid.uuid4().hex[:8]}"

    def _clip(self, text):
        if len(text) <= self.max_output:
            return text
        half = self.max_output // 2
        return text[:half] + f"\n[... {len(text) - self.max_output} chars truncated ...]\n" + text[-half:]

    async def _run(self, *args, data=None, timeout=None, clip=True):
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(None if data is None else data.encode()), timeout=timeout)
        except TimeoutError:
            proc.kill()
            out, _ = await proc.communicate()
            text = out.decode("utf-8", "replace") + "\nCommand timed out"
            return 124, self._clip(text) if clip else text
        text = out.decode("utf-8", "replace")
        return proc.returncode, self._clip(text) if clip else text

    async def start(self):
        rc, _ = await self._run("docker", "image", "inspect", self.image, timeout=30)
        if rc:
            rc, out = await self._run("docker", "pull", self.image, timeout=3600)
            if rc:
                raise RuntimeError(out)
        create_args = [
            "docker", "create", "--name", self.name,
            "--network", self.network, "--hostname", "sandbox",
            "-e", "TZ=UTC", "-e", "PYTHONHASHSEED=0", "-e", "LANG=C.UTF-8",
        ]
        if self.harden:
            create_args += [
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
            ]
        if self.pids is not None:
            create_args += ["--pids-limit", str(self.pids)]
        if self.cpus is not None:
            create_args += ["--cpus", str(self.cpus)]
        if self.memory is not None:
            create_args += ["--memory", self.memory]
        create_args += self.docker_args
        create_args += [
            "--entrypoint", "/bin/bash", self.image,
            "-lc", "trap : TERM INT; sleep infinity & wait",
        ]
        rc, out = await self._run(*create_args, timeout=120)
        if rc:
            raise RuntimeError(out)
        rc, out = await self._run("docker", "start", self.name, timeout=120)
        if rc:
            await self.close()
            raise RuntimeError(out)
        return self

    async def close(self):
        await self._run("docker", "rm", "-f", self.name, timeout=120)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *_):
        await self.close()

    async def _helper(self, op, params):
        rc, out = await self._run(
            "docker", "exec", "-i", "-w", "/testbed", self.name,
            "python", "-c", _HELPER, op, "/testbed",
            data=json.dumps(params), timeout=self.command_timeout)
        return out.strip() if not rc else f"Tool failed (exit {rc}):\n{out.strip()}"

    async def list_directory(self, **params):
        return await self._helper("list_directory", params)

    async def read_file(self, **params):
        return await self._helper("read_file", params)

    async def write_file(self, **params):
        return await self._helper("write_file", params)

    async def edit(self, **params):
        return await self._helper("edit", params)

    async def glob(self, **params):
        return await self._helper("glob", params)

    async def grep_search(self, **params):
        return await self._helper("grep_search", params)

    async def run_shell_command(self, command, description=None, timeout=None,
                                is_background=False):
        if str(is_background).lower() in {"1", "true", "yes"}:
            return "Background commands are disabled; run a bounded foreground command instead."
        timeout = min(int(timeout or self.command_timeout), 1200)
        rc, out = await self._run(
            "docker", "exec", "-w", "/testbed", self.name,
            "/bin/bash", "-lc", command, timeout=timeout)
        return f"{out.rstrip()}\n\nExit code: {rc}".strip()

    async def patch(self):
        await self.run_shell_command("git add -N .", timeout=60)
        rc, out = await self._run(
            "docker", "exec", "-w", "/testbed", self.name,
            "git", "diff", "--binary", "--no-ext-diff", timeout=120, clip=False)
        if rc:
            raise RuntimeError(out)
        return out
