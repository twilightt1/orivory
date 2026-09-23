#!/usr/bin/env python3
"""OpenClaw auto-capture daemon — the Orivory answer to capture friction.

Watches an OpenClaw session/memory directory for new and changed Markdown
files, converts them into the OpenClaw session-JSON shape that Orivory's
``openclaw`` import adapter accepts, and POSTs them to Orivory's
``/api/v1/imports`` endpoint using an **agent token** (``oa_...``, scope
``memory:write``). Every captured file is attributed to the agent in the
access ledger — the same governance story as all other writes.

Claude-mem proved the pull for this pattern (90K+ stars on hook-based
zero-touch capture); this script is Orivory's equivalent for OpenClaw
gateways: no manual `add_memory` calls, sessions flow in automatically.

Usage (stdlib only — no dependencies):

    python3 openclaw_capture.py \
        --watch ~/.openclaw/workspace \
        --url http://localhost:8000 \
        --token oa_... \
        --interval 30

Deduplication is end-to-end: Orivory's import endpoint skips memories whose
(user, source_type, source_ref) already exist, and the daemon also keeps a
local state file (mtime + sha256 per file) so unchanged files are never
re-uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_WATCH_DIR = "~/.openclaw/workspace"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_INTERVAL = 30
MD_GLOB = "*.md"


def md_to_entries(text: str) -> list[dict[str, str]]:
    """Convert a Markdown session file into role-tagged entries.

    Heuristic (claude-mem-style): lines starting with `**User:**` /
    `**Assistant:**` (or `## User` / `## Assistant`) begin a turn; other
    lines continue the current turn. Files without any recognized turn
    markers become a single user entry (the whole file) — the fact that
    the user wrote it is still worth remembering.
    """
    entries: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role and current_lines:
            entries.append({"role": current_role, "content": "\n".join(current_lines).strip()})
        current_role, current_lines = None, []

    marker_re = re.compile(r"^\s*(?:\*\*|##)")

    def _parse_header(line: str) -> tuple[str, str] | None:
        """Return (role, inline-text) when the line opens a turn.

        Splits on the FIRST colon: ``**User:** content`` → ``("user",
        "content")``. Lines that start with a markdown marker (``**``/``##``)
        but have no colon are treated as a zero-content header line (role
        only) — this avoids the regex alternation-order traps where ``**``
        lands in the wrong capture group.
        """
        if not marker_re.match(line):
            return None
        stripped = marker_re.sub("", line, count=1).strip()
        head, sep, inline = stripped.partition(":")
        role = re.sub(r"[\*#\s-]+$", "", head).strip().lower()
        if role not in ("user", "assistant", "human", "ai", "agent"):
            return None
        canonical = "user" if role in ("user", "human") else "assistant"
        inline_text = re.sub(r"^[\*#\s]+", "", inline).strip() if sep else ""
        return canonical, inline_text

    for line in text.splitlines():
        header = _parse_header(line)
        if header is not None:
            flush()
            current_role, inline_text = header
            if inline_text:
                current_lines.append(inline_text)
            continue
        if current_role:
            current_lines.append(line)
    flush()

    if not entries and text.strip():
        entries.append({"role": "user", "content": text.strip()})
    return entries


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{stat.st_mtime_ns}:{digest}"


def load_state(state_file: Path) -> dict[str, str]:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_state(state_file: Path, state: dict[str, str]) -> None:
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True))


def to_session_payload(path: Path, text: str) -> dict:
    """Markdown file → OpenClaw session-JSON shape for the import adapter."""
    return {
        "session_id": hashlib.sha256(str(path).encode()).hexdigest()[:32],
        "title": path.stem.replace("-", " ").replace("_", " ")[:120],
        "entries": md_to_entries(text),
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)
        ),
    }


def post_import(url: str, token: str, payload: dict, timeout: int = 30) -> dict:
    """Send one session dump as MULTIPART, which is what the endpoint takes.

    `/api/v1/imports` declares `file: UploadFile = File(...)`; a JSON body
    answers 422 every time, so this daemon could never capture anything.
    """
    boundary = f"----orivory{uuid.uuid4().hex}"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; '
        b'filename="openclaw-session.json"\r\n',
        b"Content-Type: application/json\r\n\r\n",
        json.dumps(payload).encode(),
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="source_format"\r\n\r\n',
        b"openclaw",
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/imports",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def scan_and_capture(watch_dir: Path, state: dict[str, str], url: str, token: str,
                     dry_run: bool = False) -> tuple[int, int]:
    """Returns (captured, skipped) for this scan."""
    captured = skipped = 0
    seen = set(state.keys())

    for md_file in sorted(watch_dir.glob(MD_GLOB)):
        key = str(md_file)
        seen.discard(key)
        fingerprint = file_fingerprint(md_file)
        if state.get(key) == fingerprint:
            skipped += 1
            continue
        text = md_file.read_text(errors="replace")
        if not text.strip():
            # Dry runs never touch persistent state: recording a fingerprint
            # for a file we did not capture makes the next real run skip it
            # silently. (Regression: --dry-run used to poison the state file.)
            if not dry_run:
                state[key] = fingerprint
            continue
        payload = to_session_payload(md_file, text)
        if not payload["entries"]:
            if not dry_run:
                state[key] = fingerprint
            continue
        if dry_run:
            print(f"[dry-run] would capture: {key} "
                  f"({len(payload['entries'])} entries)")
            captured += 1
            continue
        try:
            summary = post_import(url, token, payload)
            state[key] = fingerprint
            captured += 1
            print(f"captured {key}: created={summary.get('created', '?')} "
                  f"skipped={summary.get('skipped_duplicates', '?')}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:200]
            print(f"FAILED {key}: HTTP {exc.code} {body}", file=sys.stderr)
        except urllib.error.URLError as exc:
            print(f"FAILED {key}: {exc.reason}", file=sys.stderr)

    # forget files that disappeared — they can neither be re-read nor POSTed,
    # so they vanish from state silently
    for key in seen:
        state.pop(key, None)
    return captured, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", default=DEFAULT_WATCH_DIR,
                        help=f"OpenClaw session/memory directory (default: {DEFAULT_WATCH_DIR})")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"Orivory base URL (default: {DEFAULT_URL})")
    parser.add_argument("--token", default=None,
                        help="Agent token (oa_...) with memory:write scope. "
                             "Prefer the ORIVORY_TOKEN env var — argv is visible "
                             "in `ps` output and shell history.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"Seconds between scans (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--state-file", default=None,
                        help="Capture state file (default: <watch>/.orivory-capture-state.json)")
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Scan but don't POST")
    args = parser.parse_args()

    watch_dir = Path(args.watch).expanduser().resolve()
    if not watch_dir.is_dir():
        print(f"watch directory does not exist: {watch_dir}", file=sys.stderr)
        return 2
    state_file = (
        Path(args.state_file) if args.state_file
        else watch_dir / ".orivory-capture-state.json"
    )
    state = load_state(state_file)

    token = args.token or os.environ.get("ORIVORY_TOKEN")
    if not token:
        parser.error("agent token required: pass --token or set ORIVORY_TOKEN")

    print(f"watching {watch_dir} → {args.url} (interval {args.interval}s, "
          f"{'once' if args.once else 'looping'})")

    while True:
        try:
            captured, skipped = scan_and_capture(
                watch_dir, state, args.url, token, dry_run=args.dry_run
            )
            save_state(state_file, state)
            if args.once:
                print(f"done: {captured} captured, {skipped} skipped")
                return 0
            if captured or args.dry_run:
                print(f"scan: {captured} captured, {skipped} unchanged")
        except KeyboardInterrupt:
            print("stopped")
            return 0
        except Exception as exc:  # keep the daemon alive through transient errors
            print(f"scan error: {exc}", file=sys.stderr)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
