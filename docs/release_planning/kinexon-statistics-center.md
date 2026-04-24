# Kinexon Statistics Center API — Integration Plan

## Context

The Kinexon **Statistics Center API** (also called MIS — Match Information System) is a
**separate** API from the existing Kinexon Cloud API. This is the **primary production data
source** for live xG delivery — its WebSocket provides real-time shot events during matches.

| | Cloud API (existing) | Statistics Center API (new) |
|---|---|---|
| Base URL | `https://hbl-cloud.kinexon.com` | `https://hbl.kinexon.com/statistics-center/interfaces-api` |
| Auth | API key + basic auth | JWT via POST `/auth/login` |
| Position data | Yes (CSV download per session) | Yes in enriched shot events; confirm WebSocket parity live |
| Shot events | Yes (via `get_events_for_session`) | Yes (REST + **live via WebSocket**) |
| Match stats | No | Yes (`/stats/{matchId}`) |
| Games list | Indirectly (sessions per team) | Yes (`/games?season=2024_2025`) |
| **Live data** | **No** | **Yes (WebSocket at port 5002)** |
| Role in production | Training data, backfill | **Live trigger + live shot data** |

---

## API Endpoints

### Authentication

```
POST /auth/login
Content-Type: application/json

{"name": "madams@schueco.com", "password": "<password>"}

Response: {"message": "...", "jwt": "eyJhbG..."}
```

The JWT must be included in all subsequent requests:
```
Authorization: Bearer <jwt>
```

### REST Endpoints

```
GET /games?season=2024_2025
  -> [{"match_id": 1234813, "home_team": "Team A", "away_team": "Team B"}, ...]

GET /stats/{matchId}
  -> [{...}]  (player-level match stats)

GET /events/{matchId}?event=shot
  -> [{...}]  (shot/pass/jump events for the match)
```

### WebSocket — the live production interface

```
ws://hbl.kinexon.com:5002

# Connect with JWT in Authorization header (via polling transport extraHeaders)
# Then subscribe:

# Get list of today's active matches:
{"subscription": 1, "type": "matches", "identifier": "2024_2025"}

# Subscribe to live shot/goal events during a match:
{"subscription": 1, "type": "live_events", "identifier": "<MATCH_ID>"}

# Subscribe to stats updates during a match:
{"subscription": 1, "type": "stats", "identifier": "<MATCH_ID>"}

# Subscribe to all events (including shots filtered):
{"subscription": 1, "type": "events", "identifier": "<MATCH_ID>",
 "filter": {"event": "shot"}}
```

**Types:** `matches`, `stats`, `events`, `live_events`

---

## Enriched Shot Event Format — Verified

Analyzed 4 sample matches (580 events) from Christian Hülsemeyer (Kinexon). The enriched
shot events contain **full player positions**, making live xG with the full feature set possible.

Sample files: `data/compare/shots-*.json`

### Event structure

```json
{
  "timestamp": "2025-10-16 18:30:24",
  "timestamp_ms": 1760639424544,
  "game_clock": "56:54",
  "player_id": 2295,
  "league_id": 1515725,
  "distance": 7.570273,
  "speed_ball": 26.248363,
  "shot_position_x": -12.429924,
  "shot_position_y": 0.054619655,
  "hit_position_y": -0.49724376,
  "hit_position_z": 1.1130021,
  "trajectory": "-12.43,0.05;-12.43,0.05",
  "success": 1,
  "shot_category": "penalty",
  "goalkeeper_id": 2029,
  "goalkeeper_league_id": 1559140,
  "team": "Rhein-Neckar Löwen",
  "team_id": 4004,
  "last_group": 2,
  "match_id": "62089910",
  "id": 21808215,
  "event_type": "detected_shot_handball",
  "event": "shot",
  "validated": 1,
  "player_positions": [
    {"player_id": 65, "league_id": 355978, "x": 30.836, "y": 6.964},
    {"player_id": 100, "league_id": 873920, "x": 7.369, "y": 13.178},
    ...
  ]
}
```

### Key fields for xG feature computation

| Field | Fill rate | Maps to |
|-------|-----------|---------|
| `player_positions` (non-empty) | 513/580 (88%) | All position-based features (14 players x/y) |
| `distance` | 554/580 (96%) | `shooter_distance_to_goal` |
| `speed_ball` | 551/580 (95%) | Ball speed feature |
| `shot_position_x`, `shot_position_y` | 554/580 (96%) | `shot_angle_to_goal`, shooter lateral offset |
| `goalkeeper_league_id` | 441/580 (76%) | Identify GK in positions for `gk_distance_to_goal` etc. |
| `team`, `team_id` | 580/580 (100%) | Identify offense/defense in positions |
| `league_id` (shooter) | 580/580 (100%) | Identify shooter in positions |
| `match_id` | 580/580 (100%) | Fixture identification |
| `success` | 580/580 (100%) | Target variable |

### Position array details

- Typically **14 entries** per event (min 12, max 15, avg 13.9)
- Each entry: `{player_id, league_id, x, y}` — no `group_name` (team assignment must be
  inferred from roster or the event's `team` field)
- **Shooter is present** in positions (verified)
- **Goalkeeper is present** in positions (verified)
- **No ball position** in the array — use `shot_position_x/y` instead
- ~12% of events have empty positions — need fallback strategy

### Open items

- `player_positions` has no `group_name` — we need to map `league_id` to team. Use pre-loaded
  roster from `/stats/{matchId}` or `players` table to assign offense/defense.
- 12% empty positions: use shot-level features only (distance, angle, speed) or skip.
- Verify that the WebSocket delivers the same enriched format as the REST API test files.

### Concrete decisions for implementation

- Do not build a second reduced-feature model by default. Reuse the existing snapshot xG model.
- Build live features by converting the enriched event into minimal `df_shot_events` and
  `df_positions_normalized` frames, then call `calculate_xg_features(...)` from the existing
  pipeline code. This avoids duplicating geometry logic.
- Pre-load a `league_id -> team_name` roster map before match start so `player_positions` can be
  split into offense and defense even though the live payload has no `group_name`.
- For events with empty `player_positions`, still process the shot if core shot fields exist
  (`distance`, `shot_position_x/y`, `speed_ball`, `success`) and let missing geometry fields stay
  `NaN`; XGBoost can handle missing numeric values. Only skip if key identifiers are missing.

---

## Implementation Plan

### Phase 1: REST client (Week 1)

New file: `src/fetcher_kinexon/fetch_statistics_center.py`

```python
"""Kinexon Statistics Center API client (MIS).

Provides REST + WebSocket access for live match data.
"""

import json
import logging
import os
from datetime import datetime

import requests
import socketio

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://hbl.kinexon.com/statistics-center/interfaces-api"


class KinexonStatisticsCenterClient:
    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("KINEXON_MIS_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.username = username or os.environ["KINEXON_MIS_USERNAME"]
        self.password = password or os.environ["KINEXON_MIS_PASSWORD"]
        self._jwt: str | None = None

    def _login(self) -> str:
        resp = requests.post(
            f"{self.base_url}/auth/login",
            json={"name": self.username, "password": self.password},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("jwt") or data[0].get("jwt")
        self._jwt = token
        return token

    @property
    def jwt(self) -> str:
        if self._jwt is None:
            self._login()
        return self._jwt

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.jwt}"}

    def _get_with_retry(self, url: str, params: dict | None = None) -> requests.Response:
        resp = requests.get(url, headers=self._headers(), params=params, timeout=15)
        if resp.status_code == 401:
            self._login()
            resp = requests.get(url, headers=self._headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp

    def get_games(self, season: str = "") -> list[dict]:
        params = {"season": season} if season else {}
        return self._get_with_retry(f"{self.base_url}/games", params).json()

    def get_stats(self, match_id: str | int) -> list[dict]:
        return self._get_with_retry(f"{self.base_url}/stats/{match_id}").json()

    def get_events(self, match_id: str | int, event_type: str = "") -> list[dict]:
        params = {"event": event_type} if event_type else {}
        return self._get_with_retry(f"{self.base_url}/events/{match_id}", params).json()

    def connect_websocket(
        self,
        match_id: str | int,
        on_shot_event: callable,
        subscribe_types: list[str] | None = None,
    ) -> socketio.Client:
        """Connect to the MIS WebSocket and subscribe to live events for a match.

        Args:
            match_id: Kinexon match ID.
            on_shot_event: Callback receiving parsed shot event dict.
            subscribe_types: Which topics to subscribe to. Default: live_events + events.

        Returns:
            Connected socketio.Client (call .wait() to block, .disconnect() to stop).
        """
        subscribe_types = subscribe_types or ["live_events", "events"]
        sio = socketio.Client(reconnection=True, reconnection_attempts=10)

        @sio.on("message")
        def handle_message(data):
            if isinstance(data, str):
                if data == "Connected":
                    logger.info("WebSocket authenticated, subscribing to match %s", match_id)
                    for sub_type in subscribe_types:
                        msg = {"subscription": 1, "type": sub_type, "identifier": str(match_id)}
                        if sub_type == "events":
                            msg["filter"] = {"event": "shot"}
                        sio.send(json.dumps(msg))
                    return
                data = json.loads(data)

            if isinstance(data, dict):
                event_type = data.get("type", "")
                if event_type in ("live_events", "events"):
                    on_shot_event(data)

        @sio.on("connect_error")
        def on_error(error):
            logger.error("WebSocket error: %s", error)

        @sio.on("disconnect")
        def on_disconnect():
            logger.warning("WebSocket disconnected")

        sio.connect(
            self._websocket_url(),
            transports=["polling", "websocket"],
            headers={"Authorization": f"Bearer {self.jwt}"},
        )
        return sio

    def _websocket_url(self) -> str:
        # Extract server from base URL, use port 5002
        import re
        match = re.search(r"https?://([^:/]+)", self.base_url)
        server = match.group(1) if match else "hbl.kinexon.com"
        return f"ws://{server}:5002"
```

### Phase 2: Live match runner (Week 2)

See `hosting-deployment.md` for the `scripts/run_live_match.py` implementation.

### New env vars

```bash
# Kinexon Statistics Center / MIS
KINEXON_MIS_URL="https://hbl.kinexon.com/statistics-center/interfaces-api"
KINEXON_MIS_USERNAME="madams@schueco.com"
KINEXON_MIS_PASSWORD=""
```

---

## Priority & Timeline

| Task | When | Needed for |
|------|------|-----------|
| ~~Analyze enriched event format~~ | Done | Format verified from `data/compare/` samples |
| Build `KinexonStatisticsCenterClient` (REST + WebSocket) | Week 1 (Apr 30) | REST testing, game listing, live connection |
| Build live feature extractor (`player_positions` -> `features_xg` columns) | Week 1 (Apr 28) | Live xG computation |
| Test WebSocket during real match (confirm same format as REST samples) | Week 2 (May 02) | Verify live delivery matches sample files |
| Build `scripts/run_live_match.py` | Week 2 (May 02) | Live xG loop |
| Deploy as systemd service on IONOS | Week 3 (May 13) | Production |
