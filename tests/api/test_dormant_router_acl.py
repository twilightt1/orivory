"""The namespace fence over every memory read in ``app/`` (P4a Task 5).

The dormant routers (``entities``, ``discovery``, ``insights``, ...) are the
class of surface Task 2 missed: they hold ``select(Memory)`` statements no test
was exercising, so nothing failed when they read across the namespace boundary.
This is the pin against that class — an AST scan of EVERY module under ``app/``,
allowlisted by FILE **and** statement count, so a new read, a new file, or a
statement that lost its ``namespace_predicate`` fails here, whichever router it
lives in.

The scan is statement-level (a reader may hoist the predicate into a local and
compose it into the statement's ``.where(...)``) and alias-aware: ``select(m)``
over ``m = aliased(Memory)`` is a memory read — the Task 2 pin matched only the
literal name ``Memory`` and therefore saw zero statements for the aliased form
(review M2), which is exactly how a reader could have slipped past it.

``INVENTORY`` is the review: a file listed here was read statement by statement,
and the count is what makes a silent addition impossible. ``UNGUARDED_OK`` holds
the statements whose read is DELIBERATELY unpredicated, each pinned by count:

- ingestion / import write-path lookups key on the pair they own
  (``source_ref`` + owner, ``user_id`` + ``source_type`` + ``source_ref``): the
  row set is the write's own projection, not a read of the user's library;
- the erasure residual COUNTS in ``erasure_service``: a surviving child row in
  another namespace (or another user's) MUST still be counted — a namespace
  predicate there would turn a residual into a clean-looking receipt.

Primary-key surfaces (``db.get`` + an ownership check on the loaded row) cannot
carry a predicate and are pinned behaviourally, not here.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"

# Every file under ``app/`` that reads ``memories``, with the number of such
# statements it holds. Red on a new file (review it) or a changed count.
INVENTORY = {
    "app/api/v1/discovery.py": 6,
    "app/api/v1/entities.py": 1,
    "app/api/v1/insights.py": 2,
    "app/api/v1/memories.py": 6,
    "app/ingestion/dispatcher.py": 1,
    "app/ingestion/document_memory.py": 2,
    "app/ingestion/pipeline.py": 1,
    "app/mcp_hub/tools.py": 8,
    "app/retrieval/memory/context.py": 3,
    "app/retrieval/memory/correction.py": 3,
    "app/retrieval/memory/lexical_index.py": 1,
    "app/retrieval/memory/reindex.py": 1,
    "app/retrieval/memory/retriever.py": 1,
    "app/retrieval/memory/salience.py": 1,
    "app/services/demo_data_service.py": 1,
    "app/services/digest_service.py": 3,
    "app/services/erasure_service.py": 3,
    "app/services/import_service.py": 1,
}

# file -> how many of its statements may go without the namespace predicate
# (reasons in the module docstring; every other statement must carry it).
UNGUARDED_OK = {
    "app/ingestion/dispatcher.py": 1,
    "app/ingestion/document_memory.py": 2,
    "app/ingestion/pipeline.py": 1,
    "app/services/erasure_service.py": 2,
    "app/services/import_service.py": 1,
}

# The scan must never pass vacuously: fewer files than this means the AST walk
# (or the working directory) broke, not that ``app/`` stopped reading memories.
MIN_FILES = 12

DORMANT_ROUTERS = (
    "app/api/v1/entities.py",
    "app/api/v1/discovery.py",
    "app/api/v1/insights.py",
)


def _is_select(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "select")


def _aliased_memory_names(tree: ast.AST) -> set[str]:
    """Names a module binds to an ``aliased(Memory)`` call (review M2)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Name) and func.id == "aliased"):
            continue
        if not any(isinstance(sub, ast.Name) and sub.id == "Memory"
                   for sub in ast.walk(node.value)):
            continue
        names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _namespace_aliases(tree: ast.AST) -> set[str]:
    """Names a module binds to a ``namespace_predicate(...)`` call.

    A reader may hoist the predicate into a local (``namespace = namespace_predicate(ns)``)
    and compose it into every query of the function; the alias counts as the
    predicate ONLY where a statement actually references it — a query that
    forgets the name still fails below.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Name) and func.id == "namespace_predicate"):
            continue
        aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return aliases


def _memory_statements(path: Path) -> list[tuple[int, str, bool]]:
    """``(line, source, carries_namespace_predicate)`` per memory read in ``path``.

    A read is a ``select(...)`` whose node mentions ``Memory`` (the model or a
    column of it) or an ``aliased(Memory)`` alias. The predicate may be chained
    after the ``select()``, so only the statement as a whole is judged; comments
    are stripped so a sentence cannot satisfy the pin.
    """
    source = path.read_text()
    tree = ast.parse(source)
    memory_aliases = _aliased_memory_names(tree)
    predicate_aliases = _namespace_aliases(tree)
    parents = {child: parent
               for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if not _is_select(node):
            continue
        mentions = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}
        if "Memory" not in mentions and not (mentions & memory_aliases):
            continue
        stmt: ast.AST = node
        while not isinstance(stmt, ast.stmt):
            stmt = parents[stmt]
        segment = "\n".join(line.split("#", 1)[0]
                            for line in (ast.get_source_segment(source, stmt) or "").splitlines())
        referenced = {sub.id for sub in ast.walk(stmt) if isinstance(sub, ast.Name)}
        guarded = "namespace_predicate(" in segment or bool(referenced & predicate_aliases)
        found.append((stmt.lineno, segment, guarded))
    return found


def _scan() -> dict[str, list[tuple[int, str, bool]]]:
    """Every ``app/`` module that reads ``memories``, by repo-relative path."""
    return {str(path.relative_to(REPO)): _memory_statements(path)
            for path in sorted(APP.rglob("*.py"))
            if _memory_statements(path)}


def test_every_memory_read_in_app_carries_the_namespace_predicate():
    """A ``select(Memory)`` without the boundary is a cross-namespace read.

    The inventory is the allowlist: unknown file → red (review it), count drift
    → red (a read appeared or vanished), an unpredicated statement where
    ``UNGUARDED_OK`` allows none → red.
    """
    scanned = _scan()

    assert len(scanned) >= MIN_FILES, f"only {len(scanned)} files scanned — the walk is broken"
    assert all(scanned.values()), "a scanned file has no reads: the scan is vacuous"
    assert set(DORMANT_ROUTERS) <= set(scanned), "the dormant routers must be in the scan"

    unknown = sorted(set(scanned) - set(INVENTORY))
    assert not unknown, f"new memory readers under app/ — review, then add to INVENTORY: {unknown}"

    counts = {path: len(statements) for path, statements in scanned.items()}
    drifted = {path: (counts[path], INVENTORY[path])
               for path in counts if counts[path] != INVENTORY[path]}
    assert not drifted, f"statement count drift (found, expected): {drifted}"

    unguarded = {path: [line for line, _segment, guarded in statements if not guarded]
                 for path, statements in scanned.items()}
    leaked = {path: lines for path, lines in unguarded.items()
              if len(lines) > UNGUARDED_OK.get(path, 0)}
    assert not leaked, (
        "memory reads without the namespace predicate:\n"
        + "\n".join(f"{path}:{lines}" for path, lines in leaked.items())
    )


def test_the_unguarded_statements_are_exactly_the_ones_exempted():
    """Both directions: a new exemption must be reviewed, and a fixed file must
    drop its exemption entry."""
    scanned = _scan()
    found = {path: len([1 for _l, _s, guarded in statements if not guarded])
             for path, statements in scanned.items() if any(not g for _l, _s, g in statements)}

    assert found == UNGUARDED_OK, (
        f"unguarded statements (found, exempted): {found} vs {UNGUARDED_OK}")


def test_the_scanner_catches_an_aliased_memory_select(tmp_path):
    """Review M2: ``aliased(Memory)`` is the same read as ``Memory`` — the Task 2
    pin's literal-name match saw zero statements for it."""
    unguarded = tmp_path / "aliased_reader.py"
    unguarded.write_text(
        "from sqlalchemy import select\n"
        "from sqlalchemy.orm import aliased\n"
        "from app.models.memory import Memory\n"
        "\n"
        "def read(db, user_id):\n"
        "    m = aliased(Memory)\n"
        "    return db.execute(select(m).where(m.user_id == user_id)).scalars().all()\n"
    )
    found = _memory_statements(unguarded)

    assert [(guarded) for _line, _segment, guarded in found] == [False], (
        "an aliased(Memory) select with no predicate must be a flagged read — not an invisible one"
    )

    guarded = tmp_path / "aliased_reader_guarded.py"
    guarded.write_text(
        "from sqlalchemy import select\n"
        "from sqlalchemy.orm import aliased\n"
        "from app.models.memory import Memory\n"
        "from app.retrieval.memory.namespaces import personal_namespace\n"
        "from app.retrieval.memory.visibility import namespace_predicate\n"
        "\n"
        "def read(db, user_id):\n"
        "    m = aliased(Memory)\n"
        "    return db.execute(select(m).where(m.user_id == user_id,\n"
        "        namespace_predicate(personal_namespace(user_id)))).scalars().all()\n"
    )
    assert [g for _l, _s, g in _memory_statements(guarded)] == [True]
