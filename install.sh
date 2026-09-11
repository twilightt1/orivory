#!/usr/bin/env bash
# Orivory one-command installer — the `npx claude-mem install` equivalent.
#
# Bootstraps the lite deployment (single container: SQLite + in-process
# Chroma), creates the account + agent token, and proves first recall.
# Needs curl + docker + python3 (JSON parsing only); everything idempotent.
#
#   curl -fsSL https://raw.githubusercontent.com/twilightt1/orivory/main/install.sh | bash
#
# Flags: --port 8000  --dir ~/.orivory  --with-capture --agent-name NAME
set -euo pipefail

PORT="8000"
DIR="${HOME}/.orivory"
IMAGE="ghcr.io/twilightt1/orivory:lite"
AGENT_NAME="openclaw-capture"
WITH_CAPTURE=0

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --agent-name) AGENT_NAME="$2"; shift 2 ;;
    --with-capture) WITH_CAPTURE=1; shift ;;
    -h|--help)
      sed -n '2,9p' "$0"; exit 0 ;;
    *) warn "unknown flag: $1"; exit 2 ;;
  esac
done

command -v docker >/dev/null 2>&1 || { warn "docker is required: https://docs.docker.com/get-docker/"; exit 1; }

mkdir -p "$DIR"

say "pulling $IMAGE"
docker pull -q "$IMAGE" >/dev/null

say "starting Orivory (lite) on :$PORT — data in $DIR"
docker rm -f orivory-lite >/dev/null 2>&1 || true
docker run -d --name orivory-lite \
  -p "$PORT":8000 \
  -v "$DIR/data:/data" \
  --restart unless-stopped \
  -e LITE_MODE=1 \
  "$IMAGE" >/dev/null

say "waiting for health"
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 \
  || { warn "service did not become healthy — check: docker logs orivory-lite"; exit 1; }
say "healthy: http://localhost:$PORT"

API="http://localhost:$PORT/api/v1"
EMAIL="you@example.com"
PASS="change-me-12345"

say "creating your account + agent token (copy-pasteable afterwards)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}")
case "$CODE" in
  201) say "account created" ;;
  409) say "account exists, reusing it" ;;
  *) warn "register failed (HTTP $CODE)"; exit 1 ;;
esac
TOKEN=$(curl -fsS -X POST "$API/auth/login" -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -fsS -X POST "$API/auth/onboarding" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"display_name":"local user"}' >/dev/null
AGENT_TOKEN=$(curl -fsS -X POST "$API/agents" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"quickstart","scopes":["memory:read","memory:write"]}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

say "storing + recalling your first memory (zero API keys needed)"
curl -fsS -X POST "$API/memories" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"content":"Orivory remembers this without any API keys."}' >/dev/null
RECALL=$(curl -fsS -X POST "$API/memories/recall" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"query":"what does Orivory remember?"}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['results'][0]['content'] if d['results'] else 'NO-RECALL')")
say "recalled: $RECALL"

if [[ "$WITH_CAPTURE" == 1 ]]; then
  say "auto-capture wiring (needs one repo clone for the script)"
  echo "  git clone https://github.com/twilightt1/orivory.git && \\"
  echo "  python3 orivory/scripts/openclaw_capture.py --watch ~/.openclaw/workspace \\"
  echo "    --url http://localhost:$PORT --token $AGENT_TOKEN --interval 30"
fi

cat <<EOF

  Orivory (lite) is running — account, agent token, and first recall done.

    API       : http://localhost:$PORT  (JSON only, no web UI in lite)
    Health    : http://localhost:$PORT/health
    Data      : $DIR/data
    MCP       : http://localhost:$PORT/mcp
    Agent key : $AGENT_TOKEN  (shown once — save it)

  Your login: $EMAIL / $PASS  (change it: POST /api/v1/users/me/change-password)

  Point any MCP agent at the endpoint above with the agent key:
    {"mcp":{"servers":{"orivory":{"url":"http://localhost:$PORT/mcp",
      "transport":"streamable-http",
      "headers":{"Authorization":"Bearer $AGENT_TOKEN"}}}}}

  Next: rerun this script any time (idempotent) or add keys for better
  recall: docker rm -f orivory-lite, then add -e JINA_API_KEY=... and rerun.

  Manage: docker logs -f orivory-lite | docker restart orivory-lite |
          docker rm -f orivory-lite   (stop)

EOF
