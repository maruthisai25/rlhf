"""Per-episode bash sandbox.

Each episode gets a fresh directory under SANDBOX_ROOT. Commands are executed with
`bash -c` inside that directory with a wall-clock timeout, a lowered ulimit, a scrubbed
environment (HOME points at the sandbox), and a denylist for obviously destructive
commands. This is *not* a security boundary; it is a guard rail so that a 0.8B model
exploring randomly cannot trash the WSL distro by accident.

Must run on Linux (WSL Ubuntu). On Windows, run everything through `wsl -d Ubuntu`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass

SANDBOX_ROOT = os.environ.get("TERMBENCH_SANDBOX_ROOT", "/tmp/termbench")
DEFAULT_TIMEOUT = float(os.environ.get("TERMBENCH_CMD_TIMEOUT", "10"))
MAX_OUTPUT_CHARS = int(os.environ.get("TERMBENCH_MAX_OUTPUT", "1500"))

# Patterns that are never worth letting a tiny model try. Matched against the raw command.
_DENY = [
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+)?(-[a-zA-Z]*\s+)*/(\s|$|\*)",  # rm -rf / or rm -rf /*
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+(-[a-zA-Z]*\s+)*~(/|\s|$)",     # rm -rf ~
    r"\bmkfs(\.|\s)",
    r"\bdd\s+.*of=/dev/",
    r">\s*/dev/sd",
    r":\(\)\s*\{\s*:\|:&\s*\};:",  # fork bomb
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r"\bsudo\b",
    r"\bchmod\s+(-R\s+)?[0-7]*\s+/(\s|$)",
    r"\bcurl\b|\bwget\b|\bapt(-get)?\b|\bpip3?\s+install\b",  # no network / installs
    # tasks never need to touch anything outside the working dir: refuse rm/mv/find-delete on
    # absolute paths, parent traversal, or $HOME-style targets, and any `find /`.
    r"\brm\s+[^|;&]*(\s|=)(/|~|\.\.)",
    r"\bmv\s+[^|;&]*(\s|=)(/|~)[^\s]*\s*$",
    r"\bfind\s+(/|~|\.\.)",
    r"\b(truncate|shred)\s+[^|;&]*\s/",
    r"\bchmod\s+[^|;&]*\s/",
]
_DENY_RE = [re.compile(p) for p in _DENY]


def is_denied(command: str) -> str | None:
    for rx in _DENY_RE:
        if rx.search(command):
            return f"command refused by sandbox policy (matched /{rx.pattern}/)"
    return None


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    denied: bool = False

    def observation(self, max_chars: int = MAX_OUTPUT_CHARS) -> str:
        """Render the result the way the agent sees it."""
        if self.denied:
            return f"[exit {self.exit_code}] {self.stderr}"
        out = self.stdout
        err = self.stderr
        parts = []
        if out:
            parts.append(out.rstrip("\n"))
        if err:
            parts.append("stderr: " + err.rstrip("\n"))
        body = "\n".join(parts)
        if len(body) > max_chars:
            body = body[: max_chars // 2] + "\n... [truncated] ...\n" + body[-max_chars // 2 :]
        if self.timed_out:
            body = (body + "\n" if body else "") + "[timed out]"
        return f"[exit {self.exit_code}]" + ("\n" + body if body else " (no output)")


class Sandbox:
    """A throwaway working directory plus a `run` method."""

    def __init__(self, root: str = SANDBOX_ROOT, timeout: float = DEFAULT_TIMEOUT):
        if os.name == "nt":
            raise RuntimeError(
                "termbench.sandbox must run under Linux/WSL. Launch scripts via "
                "`wsl -d Ubuntu -- bash -lc '...'` from Windows."
            )
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.timeout = timeout
        self.dir = tempfile.mkdtemp(prefix=uuid.uuid4().hex[:8] + "_", dir=root)
        self.env = {
            "HOME": self.dir,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "dumb",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def run(self, command: str, timeout: float | None = None, check_policy: bool = True) -> ExecResult:
        if check_policy:
            why = is_denied(command)
            if why:
                return ExecResult("", why, 126, denied=True)
        # ulimit: 20s cpu, 2 GB virtual mem, 64 MB files, 256 procs. Applied inside the shell.
        wrapped = "ulimit -t 20 -v 2097152 -f 65536 -u 256 2>/dev/null; " + command
        try:
            p = subprocess.run(
                ["bash", "-c", wrapped],
                cwd=self.dir,
                env=self.env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout or self.timeout,
                stdin=subprocess.DEVNULL,
            )
            return ExecResult(p.stdout, p.stderr, p.returncode)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return ExecResult(out, err, 124, timed_out=True)

    def snapshot(self, max_entries: int = 40) -> str:
        """`ls -la`-ish listing shown to the agent at episode start."""
        r = self.run("find . -maxdepth 2 -not -path './.*' | sort | head -n %d" % max_entries, check_policy=False)
        return r.stdout.strip() or "."

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def cleanup_root(root: str = SANDBOX_ROOT):
    """Remove all leftover episode dirs (call between runs)."""
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    with Sandbox() as sb:
        print("sandbox:", sb.dir)
        print(sb.run("echo hello; ls -la; pwd").observation())
        print(sb.run("sleep 30", timeout=2).observation())
        print(sb.run("rm -rf /").observation())
    print("ok", file=sys.stderr)
