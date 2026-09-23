"""The shipped installer, read as a shell would read it.

Three defects shipped in it that nothing checked, and each broke a first run:
the first-memory curl used ``$TOKEN``, which the script never assigns — with
``set -u`` that aborts every install *after* minting the agent key and before
printing it; the same script bind-mounts ``$DIR/data`` but only created
``$DIR``, so Docker created ``data/`` root-owned while the image runs as
``app``; and the prod compose volume was unnamed (project-prefixed at runtime)
while the backup runbook mounts the literal name, so backups came out empty.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALL = (ROOT / "install.sh").read_text()
# Read as text: the file carries Compose's `!override` tag, which plain YAML
# constructors refuse (the CI step validates it with `docker compose config`).
PROD = (ROOT / "docker-compose.prod.yml").read_text()


def test_the_installer_uses_no_unassigned_token_variable():
    assert "$AGENT_TOKEN" in INSTALL
    assert not re.search(r"\$\{?TOKEN\b", INSTALL), (
        "install.sh references $TOKEN, which it never assigns (only $AGENT_TOKEN)"
    )


def test_the_installer_creates_the_data_directory_it_bind_mounts():
    assert '$DIR/data' in INSTALL
    assert re.search(r'mkdir -p "\$DIR/data"', INSTALL), (
        "install.sh mounts $DIR/data but never creates it: dockerd makes it root:root "
        "and the image runs as app, so bootstrap_sqlite() cannot write"
    )


def test_the_prod_volume_is_named_what_the_runbook_mounts():
    assert re.search(r"^  orivory-data:\s*\n(?:    #.*\n)*    name: orivory-data\b", PROD, re.M), (
        "without an explicit name Compose prefixes the volume with the project, so the "
        "runbook's `-v orivory-data:/data` mounts a different, empty volume"
    )
    runbook = (ROOT / "docs" / "BACKUP_RESTORE.md").read_text()
    assert "-v orivory-data:/data" in runbook
