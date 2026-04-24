# Hosting & Deployment Plan — IONOS

## Context

HBL hosts production services via **IONOS**. The xG pipeline must run as a **persistent service**
during matches, not as a cron job. It connects to the Kinexon MIS WebSocket, listens for live
shot events, computes xG, and pushes results to Sportradar in real time.

---

## Proposed Architecture

```
IONOS Linux Server (VPS / Managed)
├── /opt/bielemetrics/
│   ├── BieLeMetrics/
│   │   ├── data/
│   │   │   ├── hbl_raw.duckdb           ← training/backfill data
│   │   │   ├── models/xg_context.joblib ← trained xG model
│   │   │   └── live/                    ← live match state + logs
│   │   ├── .env                          ← secrets (600 perms)
│   │   └── scripts/
│   │       ├── run_live_match.py         ← live service entry point
│   │       └── run_post_game.py          ← batch backfill/audit
│   └── logs/
│       └── bielemetrics.log
├── Python 3.12 + uv
└── systemd service: bielemetrics-live.service
```

---

## Operating Modes

### 1. Live Mode (production — during matches)

A **systemd service** runs `scripts/run_live_match.py` as a persistent process. On match days:

1. Fetches today's games from Kinexon MIS REST `/games`
2. Connects to Kinexon MIS WebSocket, subscribes to `live_events` for active matches
3. On each shot event: loads trained model, computes xG from shot features, updates cumulative
   per-player totals
4. Pushes updated xGoals to Sportradar webhook immediately
5. Persists shot-level results to `data/live/` for post-game audit

The service runs continuously and reconnects on WebSocket drops.

### 2. Batch Mode (offline — backfill + training)

Cron or manual execution of the existing Dagster pipeline for:
- Historical fixture backfill (full positional features)
- xG model training/retraining
- Post-game audit: compare live-pushed xG values against batch-computed values

---

## Live Match Service: `scripts/run_live_match.py`

```python
"""Live xG service — connects to Kinexon MIS WebSocket, computes xG per shot, pushes to Sportradar.

Usage:
    uv run python scripts/run_live_match.py                    # auto-detect today's matches
    uv run python scripts/run_live_match.py --match <match_id> # specific match
    uv run python scripts/run_live_match.py --dry-run          # compute but don't push
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from src.fetcher_kinexon.fetch_statistics_center import KinexonStatisticsCenterClient
from src.pipelines.features.calc_xg_features import calculate_xg_features
from src.pipelines.ml.infer_xg import infer_xg
from src.pipelines.push.push_sportradar_xg import push_xg_to_sportradar

logger = logging.getLogger("live_xg")

LIVE_STATE_DIR = Path("data/live")
MODEL_PATH = "data/models/xg_context.joblib"


class LiveXgState:
    """Tracks cumulative per-player xGoals for a single match."""

    def __init__(self, match_id: str):
        self.match_id = match_id
        self.player_xg: dict[str, float] = defaultdict(float)
        self.player_info: dict[str, dict] = {}
        self.shot_log: list[dict] = []
        self.n_shots = 0

    def add_shot(self, shot_event: dict, xg_value: float) -> None:
        player_id = str(shot_event.get("league_id", shot_event.get("player_id", "unknown")))
        self.player_xg[player_id] += xg_value
        self.n_shots += 1

        # Store player info if available in the event
        if player_id not in self.player_info:
            self.player_info[player_id] = {
                "player_id": player_id,
                "league_id": str(shot_event.get("league_id", "")),
                "team": str(shot_event.get("team", "")),
                "first_name": str(shot_event.get("first_name", "")),
                "last_name": str(shot_event.get("last_name", "")),
            }

        self.shot_log.append({
            "player_id": player_id,
            "xg": round(xg_value, 4),
            "cumulative_xg": round(self.player_xg[player_id], 4),
            "shot_number": self.n_shots,
            "raw_event": shot_event,
        })

    def build_payload(self) -> list[dict]:
        """Build Sportradar webhook payload from current state."""
        records = []
        for player_id, xg_total in self.player_xg.items():
            info = self.player_info.get(player_id, {})
            records.append({
                "match_id": self.match_id,
                "player_id": player_id,
                "league_id": info.get("league_id", ""),
                "team": info.get("team", ""),
                "first_name": info.get("first_name", ""),
                "last_name": info.get("last_name", ""),
                "xGoals": round(float(xg_total), 4),
            })
        return records

    def save_state(self) -> None:
        state_file = LIVE_STATE_DIR / f"match_{self.match_id}.json"
        LIVE_STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({
            "match_id": self.match_id,
            "n_shots": self.n_shots,
            "player_xg": dict(self.player_xg),
            "player_info": self.player_info,
            "shot_log": self.shot_log,
        }, indent=2, default=str))


def _goal_position_from_shot(shot_event: dict) -> float:
    shot_x = float(shot_event.get("shot_position_x", 0.0) or 0.0)
    return 20.0 if shot_x < 0 else -20.0


def _build_live_feature_frames(
    match_id: str,
    shot_event: dict,
    roster_map: dict[str, dict],
    match_meta: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapt one enriched live event into the existing batch feature-engineering inputs.

    The key implementation decision is to reuse `calculate_xg_features(...)` instead of
    reimplementing geometry logic in the live service.
    """
    shooter_team = shot_event["team"]
    team_names = {v.get("team_name") for v in roster_map.values() if v.get("team_name")}
    team_names.discard(None)
    defense_team = next((name for name in team_names if name != shooter_team), None)

    df_match = pd.DataFrame([
        {
            "fixture_id": match_id,
            "attendance": match_meta.get("attendance"),
        }
    ])

    df_shot = pd.DataFrame([
        {
            "fixture_id": match_id,
            "event_id": str(shot_event["id"]),
            "throw_timestamp_ms": shot_event["timestamp_ms"],
            "person_league_id": str(shot_event["league_id"]),
            "goalkeeper_league_id": str(shot_event.get("goalkeeper_league_id") or ""),
            "team_name_offense": shooter_team,
            "team_name_defense": defense_team,
            "team_name_home": match_meta.get("home_team"),
            "goal_position": _goal_position_from_shot(shot_event),
            "attack_type": shot_event.get("shot_category"),
            "sub_type": shot_event.get("event_type"),
            "success": int(shot_event["success"]),
        }
    ])

    positions = []
    for p in shot_event.get("player_positions", []):
        roster_entry = roster_map.get(str(p.get("league_id")), {})
        positions.append(
            {
                "timestamp_ms": shot_event["timestamp_ms"],
                "group_name": roster_entry.get("team_name"),
                "league_id": str(p.get("league_id")),
                "x_m": float(p.get("x")),
                "y_m": float(p.get("y")),
            }
        )
    df_positions = pd.DataFrame(
        positions,
        columns=["timestamp_ms", "group_name", "league_id", "x_m", "y_m"],
    )
    return df_match, df_shot, df_positions


def compute_xg_from_live_event(model, shot_event: dict, roster_map: dict[str, dict], match_meta: dict) -> float:
    """Compute xG by reusing the existing batch feature pipeline on one live event."""
    df_match, df_shot, df_positions = _build_live_feature_frames(
        match_id=str(shot_event["match_id"]),
        shot_event=shot_event,
        roster_map=roster_map,
        match_meta=match_meta,
    )
    df_features = calculate_xg_features(df_match, df_shot, df_positions)
    df_pred = infer_xg(model=model, df_features_xg=df_features, proba_col="xg")
    return float(df_pred["xg"].iloc[0])


def run_live_match(match_id: str, *, dry_run: bool = False) -> None:
    """Connect to MIS WebSocket and run live xG loop for one match."""
    client = KinexonStatisticsCenterClient()
    model = joblib.load(MODEL_PATH)
    state = LiveXgState(match_id)
    player_stats = client.get_stats(match_id)
    roster_map = {
        str(row["league_id"]): {
            "team_name": row.get("team"),
            "first_name": row.get("first_name"),
            "last_name": row.get("last_name"),
            "number": row.get("number"),
            "function": row.get("function"),
        }
        for row in player_stats
        if row.get("league_id") is not None
    }
    games = client.get_games(season="2024_2025")
    match_meta = next((g for g in games if str(g.get("match_id")) == str(match_id)), {})

    def on_shot_event(data: dict) -> None:
        events = data.get("data", [data])
        if not isinstance(events, list):
            events = [events]

        for event in events:
            xg = compute_xg_from_live_event(model, event, roster_map, match_meta)
            state.add_shot(event, xg)
            state.save_state()

            logger.info(
                "Shot #%d in match %s: player=%s xg=%.3f cumulative=%.3f",
                state.n_shots, match_id,
                event.get("player_id", "?"), xg,
                state.player_xg.get(str(event.get("player_id", "")), 0),
            )

            # Push updated totals to Sportradar after each shot
            payload = state.build_payload()
            if payload:
                push_xg_to_sportradar(payload, dry_run=dry_run)

    logger.info("Connecting to MIS WebSocket for match %s", match_id)
    sio = client.connect_websocket(
        match_id=match_id,
        on_shot_event=on_shot_event,
        subscribe_types=["live_events", "events"],
    )

    try:
        sio.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted — saving final state")
    finally:
        state.save_state()
        sio.disconnect()
        logger.info("Match %s finished: %d shots processed", match_id, state.n_shots)


def main():
    parser = argparse.ArgumentParser(description="Live xG service")
    parser.add_argument("--match", help="Specific match ID to track")
    parser.add_argument("--dry-run", action="store_true", help="Compute xG but don't push")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/live/live_xg.log", mode="a"),
        ],
    )

    client = KinexonStatisticsCenterClient()

    if args.match:
        run_live_match(args.match, dry_run=args.dry_run)
    else:
        # Auto-detect today's active matches
        games = client.get_games(season="2024_2025")
        today = date.today().isoformat()
        active_games = [g for g in games if g.get("date") == today]

        if not active_games:
            logger.info("No games today (%s)", today)
            return

        # For now: handle one match at a time. Multi-match needs threading.
        for game in active_games:
            match_id = str(game["match_id"])
            logger.info("Starting live xG for match %s: %s vs %s",
                        match_id, game.get("home_team"), game.get("away_team"))
            run_live_match(match_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

Notes on the implementation sketch:
- The service should key live cumulative state by `league_id`, because that identifier is always
  present in the Kinexon event payload.
- Final webhook payload enrichment still needs roster-backed fields such as Sportradar
  `player_id`, `team_id`, `number`, and `function` before production use.
- The important architectural choice is the reuse of `calculate_xg_features(...)` for live
  events instead of a separate hand-written geometry implementation.

---

## systemd Service

### `/etc/systemd/system/bielemetrics-live.service`

```ini
[Unit]
Description=BieLeMetrics Live xG Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=bielemetrics
WorkingDirectory=/opt/bielemetrics/BieLeMetrics
ExecStart=/usr/local/bin/uv run python scripts/run_live_match.py
Restart=on-failure
RestartSec=30
StandardOutput=append:/opt/bielemetrics/logs/bielemetrics.log
StandardError=append:/opt/bielemetrics/logs/bielemetrics.log
EnvironmentFile=/opt/bielemetrics/BieLeMetrics/.env

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable bielemetrics-live
systemctl start bielemetrics-live
journalctl -u bielemetrics-live -f    # follow logs
```

**Note:** The service runs continuously. On non-match days it will poll `/games`, find nothing,
and idle. On match days it connects to the WebSocket and processes shots. The `Restart=on-failure`
with `RestartSec=30` handles crashes and WebSocket disconnects.

---

## Batch Mode (cron — backfill + training)

The batch pipeline continues to run as before for historical data and model training:

```bash
# /etc/cron.d/bielemetrics-batch

# Weekly model retrain (Sunday 05:00)
0 5 * * 0 bielemetrics cd /opt/bielemetrics/BieLeMetrics && /usr/local/bin/uv run python scripts/debug/ml.py train >> /opt/bielemetrics/logs/batch.log 2>&1

# Daily DB backup at 04:00
0 4 * * * bielemetrics cp /opt/bielemetrics/BieLeMetrics/data/hbl_raw.duckdb /opt/bielemetrics/backups/hbl_raw_$(date +\%Y\%m\%d).duckdb

# Post-game audit: backfill yesterday's matches with full features (for model training data)
0 6 * * * bielemetrics cd /opt/bielemetrics/BieLeMetrics && /usr/local/bin/uv run python scripts/run_post_game.py --yesterday >> /opt/bielemetrics/logs/batch.log 2>&1
```

The post-game batch backfill ensures the DuckDB has full positional features for every match,
which feeds the next model retrain cycle.

---

## Deployment Steps

### 1. Server provisioning (Daniel / IONOS)

- **Minimum spec:** 8 GB RAM, 4 vCPU, 100 GB SSD
- **OS:** Ubuntu 22.04+ or Debian 12+
- **Network:** Outbound HTTPS + WSS to `hbl.kinexon.com`, `hbl-cloud.kinexon.com`, Sportradar APIs
- **Network (important):** Outbound WebSocket on port 5002 to `hbl.kinexon.com` must not be blocked
- **SSH access** for Michael
- **System user:** Create `bielemetrics` user for the service

### 2. Initial setup

```bash
apt update && apt install -y python3.12 python3-pip git
pip install uv

useradd -r -m -d /opt/bielemetrics bielemetrics
git clone <repo_url> /opt/bielemetrics/BieLeMetrics
chown -R bielemetrics:bielemetrics /opt/bielemetrics

cd /opt/bielemetrics/BieLeMetrics
sudo -u bielemetrics uv sync --no-dev

# .env with all secrets
cp .env.example .env && chmod 600 .env
# Edit .env with actual credentials
```

### 3. Database + model initialization

```bash
# Copy pre-built DuckDB from dev machine (positions cannot be re-fetched)
scp data/hbl_raw.duckdb ionos:/opt/bielemetrics/BieLeMetrics/data/

# Train model on server
sudo -u bielemetrics uv run python scripts/debug/ml.py train
```

### 4. Enable and start service

```bash
cp bielemetrics-live.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable bielemetrics-live
systemctl start bielemetrics-live
```

---

## Monitoring

### Minimum viable for May delivery

- **Service health:** `systemctl status bielemetrics-live`
- **Logs:** `/opt/bielemetrics/logs/bielemetrics.log` — all shots, xG values, push results
- **Live state files:** `data/live/match_<id>.json` — per-match shot log with cumulative xG
- **Email alert on service failure:** systemd `OnFailure=` directive or simple wrapper script

### Questions for 2026-05-08 meeting

- Does IONOS provide SMTP for alert emails?
- Does Luis need SSH access to check logs?
- Should there be a simple status page for Daniel?
- What is the acceptable latency between shot and xG push? (1s? 5s? 30s?)

---

## Rollback & Recovery

| Scenario | Action |
|----------|--------|
| WebSocket drops mid-game | systemd restarts service within 30s, reconnects, resumes |
| Bad xG values pushed | Re-push corrected values from post-game batch audit |
| Service crash | `systemctl restart bielemetrics-live` |
| DuckDB corruption | Restore from daily backup |
| Model quality degrades | Retrain: `uv run python scripts/debug/ml.py train` |
| Server unreachable on match day | Manual push from dev machine: `uv run python scripts/run_live_match.py --match <id>` |

---

## Cost Estimate (IONOS)

| Spec | Monthly cost |
|------|-------------|
| VPS L (8 GB RAM, 4 vCPU, 160 GB SSD) | ~15-20 EUR/month |
| VPS XL (16 GB RAM, 6 vCPU, 240 GB SSD) | ~25-30 EUR/month |

The service is mostly idle (WebSocket listener waiting for shots). CPU/RAM spikes only briefly
per shot for model inference. A basic VPS is sufficient.
