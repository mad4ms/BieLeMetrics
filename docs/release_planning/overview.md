# Release Planning: Live xG Delivery to HBL / Sportradar

## Scope

Deliver **live xG values** (expected goals per player per shot) to Sportradar via webhook
**during each match** as shots happen. The spatiotemporal sequence model is **out of scope**
for this release — we ship the snapshot XGBoost model.

Input sources:
- Sportradar DataCore REST API (match events, player roster, fixture catalog)
- Kinexon Cloud API (raw positional tracking CSVs, session metadata — for training + backfill)
- Kinexon Statistics Center API / MIS (live shot events via WebSocket — production trigger)

Output: POST updated per-player cumulative xGoals to Sportradar webhook after each shot during
a live match.

---

## Live Architecture

```
                          DURING MATCH (live)
                          ==================
Kinexon MIS WebSocket ─── live_events (enriched shot event)
        |
        v
  shot event + player_positions
  (14 players x/y, shooter position,
   speed_ball, distance, GK ID)
        |
        v
  compute xG features from positions
  (same features as batch — full feature set)
        |
        v
  update cumulative per-player xGoals
        |
        v
  POST to Sportradar webhook
        |
        v
  persist to data/live/ for audit


                          OFFLINE (batch)
                          ==================
Kinexon Cloud API ──> positions_kinexon_raw ──> features_xg
Sportradar REST   ──> fixture_events_raw    ──> ml_xg_model (train periodically)
```

### Enriched Shot Events — Verified

The Kinexon MIS enriched shot events (tested against 4 sample matches, 580 events) contain
**full player positions** at the moment of the shot. This means live mode can use the **same
feature set** as batch mode — no reduced model needed.

Per event:
- `player_positions`: array of ~14 entries, each with `{player_id, league_id, x, y}`
- Shooter and goalkeeper are present in the positions array
- `distance` (to goal), `speed_ball`, `shot_position_x/y`: 96% fill rate
- `goalkeeper_league_id`: 76% fill rate
- `team`, `team_id`, `match_id`, `league_id`: 100% fill rate
- `player_positions` non-empty: 88% of events (67 of 580 have empty positions)

Sample files: `data/compare/shots-*.json` (4 matches from Christian Hülsemeyer, Kinexon)

### Two operating modes

| Mode | When | Data source | Features available | Purpose |
|------|------|-------------|-------------------|---------|
| **Live** | During match | Kinexon MIS WebSocket `live_events` | **Full**: 14 player positions + shot metadata | Real-time xG push to Sportradar |
| **Batch** | Offline / periodic | Kinexon Cloud API + Sportradar REST | Full: 175M position rows per season | Model training, backfill, audit |

Both modes compute the same features. The live mode extracts positions from the enriched shot
event's `player_positions` array; the batch mode uses the full tracking timeline from DuckDB.

---

## API Landscape

| API | Base URL | Auth | Role |
|-----|----------|------|------|
| Sportradar DataCore REST | env `BASE_URL` | OAuth2 | Training data: events, players, teams |
| Kinexon Cloud | `https://hbl-cloud.kinexon.com` | API key + basic auth | Training data: position CSVs, sessions |
| **Kinexon Statistics Center (MIS)** | `https://hbl.kinexon.com/statistics-center/interfaces-api` | JWT | **Live production**: WebSocket for shot events during match |
| Sportradar Webhook (push) | TBD from readme | API key + HMAC | **We write**: per-player xGoals JSON |

---

## Week-by-Week Plan

### Week 1: 2026-04-25 -- 2026-05-01 (API verification + live architecture)

| Day | Task | Owner | Deliverable |
|-----|------|-------|-------------|
| Fri 25 | Re-request Sportradar DataCore credentials (Keeper expired) | Michael | Email sent |
| Fri 25 | Test Kinexon MIS: login, `/games`, `/events/{matchId}?event=shot` | Michael | Confirm REST API works with credentials |
| Mon 28 | Build live feature extractor: `player_positions` array -> `features_xg` columns | Michael | Unit test: enriched event from `data/compare/` -> same features as batch pipeline |
| Tue 29 | Build `src/pipelines/push/push_sportradar_xg.py` — payload builder + push | Michael | Unit tests pass against sample data |
| Wed 30 | Build `src/fetcher_kinexon/fetch_statistics_center.py` — REST + WebSocket client | Michael | Client connects, subscribes, receives events |
| Thu 01 | End-to-end local test: replay enriched JSON -> features -> inference -> webhook dry-run | Michael | Full pipeline works with sample data |

### Week 2: 2026-05-02 -- 2026-05-08 (live pipeline + on-site prep)

| Day | Task | Owner | Deliverable |
|-----|------|-------|-------------|
| Fri 02 | Build `scripts/run_live_match.py` — WebSocket listener + live xG loop | Michael | Connects to MIS, computes xG per shot, pushes to Sportradar (dry-run) |
| Mon 05 | Receive Sportradar webhook readme + non-prod endpoint URL | Daniel/Martin | Field names, auth, URL |
| Tue 06 | Test webhook push against Sportradar non-production endpoint | Michael | HTTP 200 for 1 shot |
| Wed 07 | Live test during a real match — full loop: MIS WebSocket -> xG -> Sportradar non-prod | Michael | xGoals pushed live for all shots in a match |
| **Thu 08** | **On-site meeting** — demo live pipeline, finalize hosting, monitoring, fallback | All | IONOS spec, alerting, reconnect policy |

### Week 3: 2026-05-09 -- 2026-05-15 (deployment + staging)

| Day | Task | Owner | Deliverable |
|-----|------|-------|-------------|
| Mon 12 | IONOS server provisioned | Daniel | SSH access for Michael |
| Tue 13 | Deploy to IONOS, configure `.env`, install as systemd service | Michael | Service starts, connects to MIS WebSocket |
| Wed 14 | Backfill historical fixtures, train model on server | Michael | `ml_xg_model` ready on IONOS |
| Thu 15 | Live match test on IONOS — full production path except webhook goes to non-prod | Michael | Automated live xG from server |

### Week 4: 2026-05-16 -- 2026-05-20 (production + handover)

| Day | Task | Owner | Deliverable |
|-----|------|-------|-------------|
| Mon 19 | Switch to Sportradar production webhook | Michael + Martin | Production xG flowing live |
| **Tue 20** | **Sportradar handover meeting** | All | Handover docs, support contacts |

---

## Open Action Items

### Credentials / Access (Updated 2026-04-24)

| Item | Owner | Status |
|------|-------|--------|
| Re-request Sportradar DataCore credentials | Michael -> support@sportradar.com | Email sent |
| Kinexon MIS API credentials in `.env` | Michael | Done |
| Kinexon Cloud credentials in `.env` | Michael | Done |
| Sportradar webhook readme (field names, auth, URL) | Daniel -> Martin Vuko | Pending |
| IONOS server provisioning | Daniel | Not started |

### Technical Work Items

| Item | Plan doc | Priority |
|------|----------|----------|
| MIS WebSocket parity check vs. enriched sample files | `kinexon-statistics-center.md` | Week 2 |
| Live feature extractor (from enriched shot event `player_positions`) | `sportradar-webhook.md` | Week 1 |
| Sportradar webhook push module | `sportradar-webhook.md` | Week 1 |
| Live match runner service (`run_live_match.py`) | `hosting-deployment.md` | Week 2 |
| IONOS deployment as systemd service | `hosting-deployment.md` | Week 3 |

### Critical Questions

1. ~~What data does `live_events` contain per shot?~~ **RESOLVED** — enriched events include 14 player positions (x, y), shot position, speed, distance. Full feature set available live.
2. Exact webhook payload field names — awaiting Sportradar readme.
3. IONOS server spec — needs to run a persistent WebSocket listener process (systemd service, not cron).
4. Reconnect strategy: what if WebSocket drops mid-game?
5. Should xG be delivered for both teams, or only the team that took the shot?
6. How to handle the ~12% of events with empty `player_positions`? Fallback to base rate? Skip?

## Release Assessment

This is now a credible release plan for the May deadline because the biggest architecture risk has
collapsed: the Kinexon enriched shot events already carry the positional context needed for the
snapshot xG feature set.

What is strong:
- Scope is now explicit: snapshot xG only, live per-shot delivery, no sequence model work
- The live data source is identified and validated with real sample payloads in `data/compare/`
- The operational split is sensible: live service for delivery, batch pipeline for retraining and audit
- Remaining blockers are integration blockers, not model-design blockers

What still decides delivery risk:
- Sportradar webhook spec and credentials are still external dependencies
- Live WebSocket payload must still be confirmed to match the sample JSON shape
- A fallback policy for the ~12% of shots without `player_positions` must be agreed before go-live

---

## Contacts

| Role | Person | Contact |
|------|--------|---------|
| HBL project lead | Daniel Koenen | koenen@daikin-hbl.de |
| HBL data support (Werkstudent) | Luis Endler | endler@daikin-hbl.de |
| Kinexon project manager | Benita Oberhofer | benita.oberhofer@kinexon.com, +49 151/46396030 |
| Sportradar client support | Martin Vuko | m.vuko.ext@sportradar.com |
| Sportradar tech lead | Chris Morgan | c.morgan@sportradar.com |
| Sportradar support (credential requests) | — | support@sportradar.com (Subject: "DataCore Api support") |
