# Sportradar Webhook Push — Technical Plan

## What needs to be built

A module `src/pipelines/push/push_sportradar_xg.py` that:

1. Takes a shot event (from Kinexon MIS WebSocket `live_events`)
2. Computes xG for the shot using the trained model
3. Updates cumulative per-player xGoals for the current match
4. Formats the payload to match the Sportradar webhook spec
5. POSTs to the Sportradar HTTPS endpoint with the required auth headers
6. Logs the response; retries on transient failures

This is called by `scripts/run_live_match.py` after every shot event during a live match.
There is no Dagster asset for the live push — it runs outside Dagster as a standalone service.

---

## Data Flow (Live Mode)

```
Kinexon MIS WebSocket
  |  live_events: shot detected
  v
shot event data
  |  extract features (position, speed, distance, angle)
  v
ml_xg_model (XGBoost, pre-trained, loaded once at service start)
  |  predict xG for this shot
  v
LiveXgState (in-memory per-match tracker)
  |  update cumulative per-player xGoals
  v
build_xg_webhook_payload()
  |  format as Sportradar JSON array
  v
push_xg_to_sportradar()
  |  POST to webhook with auth headers
  v
Sportradar DataCore ingestion
```

Each push sends the **full cumulative player totals** for the match so far, not just the
delta from the last shot. This makes the push idempotent — Sportradar can always overwrite
with the latest values.

---

## Payload Format (from Sportradar spec)

```json
[
  {
    "team": "HC Erlangen",
    "team_id": 6288,
    "match_id": "42307987",
    "player_id": 3,
    "league_id": "124930",
    "last_group": 2,
    "number": 33,
    "function": "RL",
    "first_name": "Nikolai",
    "last_name": "Link",
    "xGoals": 1.95
  }
]
```

### Field Mapping

| Webhook field | Source | Notes |
|---------------|--------|-------|
| `match_id` | Kinexon MIS `live_events` | Match identifier — **verify format maps to Sportradar fixture ID** |
| `player_id` | Kinexon shot event | May need `_extract_numeric_id()` if URN format |
| `league_id` | Kinexon shot event | Kinexon-side player ID |
| `team` | Kinexon shot event or MIS `/games` | Team name |
| `team_id` | Pre-loaded from `teams_sportradar_raw` | **Confirm format with Sportradar** |
| `first_name` | Pre-loaded from `match_players_normalized` or MIS `/stats` | |
| `last_name` | Pre-loaded from `match_players_normalized` or MIS `/stats` | |
| `function` | Pre-loaded player roster | Position (e.g. "RL", "LW") |
| `number` | Pre-loaded player roster | Jersey number — **may not exist yet** |
| `last_group` | Unknown | **Clarify from webhook readme** |
| `xGoals` | Cumulative `sum(xg)` per player for this match | Updated after every shot |

### Player Roster Pre-Loading

At match start (before subscribing to `live_events`), the service pre-loads player roster data:

```python
# Option A: From DuckDB (if fixture was already backfilled)
df_players = con.execute(
    "SELECT * FROM players WHERE fixture_id = ?", [fixture_id]
).df()

# Option B: From Kinexon MIS REST (available for active matches)
player_stats = client.get_stats(match_id)

# Option C: From Sportradar DataCore REST
# (requires DataCore credentials to be working)
```

This ensures `first_name`, `last_name`, `team`, `team_id`, etc. are available when building
the webhook payload, without depending on the shot event containing all roster fields.

### Concrete payload strategy

- Push after every shot event.
- Each push contains the full cumulative per-player xGoals for the match so far.
- Keep one in-memory match state keyed by `match_id` and overwrite totals on every push.
- If Sportradar accepts idempotent overwrites, no separate deduplication protocol is needed.
- If a single push fails, keep the updated in-memory state and retry on the next push as well.

---

## Auth Headers

```
x-organization: <org_id>      # e.g. "h1n81"
x-sport: h
x-payload-type: stats
x-signature: <signature>      # HMAC-SHA256 or static — confirm from readme
x-api-key: <api_key>
host: <webhook_host>
```

Env vars:

```bash
SPORTRADAR_WEBHOOK_URL=""
SPORTRADAR_WEBHOOK_URL_NONPROD=""
SPORTRADAR_WEBHOOK_ORG=""
SPORTRADAR_WEBHOOK_API_KEY=""
SPORTRADAR_WEBHOOK_SECRET=""
```

---

## Implementation

### `src/pipelines/push/push_sportradar_xg.py`

```python
"""Push per-player xGoals to the Sportradar DataCore webhook."""

import hashlib
import hmac
import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


def push_xg_to_sportradar(
    payload: list[dict],
    *,
    use_nonprod: bool = False,
    dry_run: bool = False,
) -> requests.Response | None:
    """POST the xGoals payload to the Sportradar webhook.

    Sends the full cumulative per-player xGoals array. Sportradar overwrites
    previous values for the same match_id, making this idempotent.

    Args:
        payload: List of player-level xGoals dicts.
        use_nonprod: If True, use SPORTRADAR_WEBHOOK_URL_NONPROD.
        dry_run: If True, log the payload but do not send.
    """
    if not payload:
        logger.debug("Empty payload, skipping push")
        return None

    url_key = "SPORTRADAR_WEBHOOK_URL_NONPROD" if use_nonprod else "SPORTRADAR_WEBHOOK_URL"
    url = os.environ[url_key]
    api_key = os.environ["SPORTRADAR_WEBHOOK_API_KEY"]
    org = os.environ["SPORTRADAR_WEBHOOK_ORG"]
    secret = os.environ.get("SPORTRADAR_WEBHOOK_SECRET", "")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest() if secret else ""

    headers = {
        "Content-Type": "application/json",
        "x-organization": org,
        "x-sport": "h",
        "x-payload-type": "stats",
        "x-signature": signature,
        "x-api-key": api_key,
    }

    if dry_run:
        logger.info("DRY RUN — would POST %d players to %s", len(payload), url)
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, data=body, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info("Pushed %d players to %s: %d", len(payload), url, response.status_code)
            return response
        except requests.RequestException as exc:
            logger.warning("Push attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)

    logger.error("All %d push attempts failed for %s", MAX_RETRIES, url)
    return None
```

---

## Testing Plan

| Phase | What | When |
|-------|------|------|
| 1. Unit test | Payload formatting with mock shot events | Week 1 |
| 2. Replay test | Replay `data/compare/shots-*.json` through live pipeline, `--dry-run`, inspect payloads | Week 1 |
| 3. Non-prod | Push to Sportradar non-production endpoint during a live match | Week 2 (after receiving webhook URL) |
| 4. Production | Push to Sportradar production endpoint | Week 4 |

### Debug commands

```bash
# Watch live shot events without pushing:
uv run python scripts/run_live_match.py --match <id> --dry-run

# Push historical fixture from DuckDB (batch mode, for testing payload format):
uv run python scripts/debug/ml.py push <fixture_id> --dry-run
uv run python scripts/debug/ml.py push <fixture_id> --nonprod
```

---

## Latency Budget

| Step | Expected time |
|------|--------------|
| Kinexon shot detection -> WebSocket delivery | ~1-2s (Kinexon internal) |
| WebSocket message parsing | <10ms |
| xG model inference (1 shot, XGBoost) | <50ms |
| Sportradar webhook POST | ~100-500ms |
| **Total: shot to Sportradar ingestion** | **~2-3 seconds** |

---

## Risks & Blockers

| Risk | Severity | Status | Mitigation |
|------|----------|--------|-----------|
| Sportradar webhook readme not yet received | **High** | Blocked | Daniel to follow up with Martin Vuko |
| ~~`live_events` payload structure unknown~~ | ~~High~~ | **Resolved** | Enriched events verified from `data/compare/` — 580 events, 4 matches |
| ~~Feature gap: live events may lack full position data~~ | ~~High~~ | **Resolved** | Enriched events contain `player_positions` (~14 players x/y) — full feature set available |
| ~12% of shots have empty `player_positions` | Medium | Known | Reuse model with missing geometry features left `NaN`; only skip if core IDs/shot fields are missing |
| `player_id`/`team_id` format mismatch (URN vs numeric) | Medium | Unknown | Add `_extract_numeric_id()`, verify against Sportradar |
| `x-signature` auth pattern unclear | Medium | Unknown | Implement HMAC, confirm from readme |
| Sportradar DataCore credentials expired | **High** | Re-requested | Blocks non-prod testing |
| WebSocket connection stability over 2h match | Medium | Unknown | Reconnect logic in socketio client |
