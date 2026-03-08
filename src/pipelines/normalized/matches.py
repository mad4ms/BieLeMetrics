import json
import difflib
import logging
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_TEAM_NAME_MAPPING = {
    "Handball Sport Verein Hamburg": "HSV Hamburg",
    "Minden": "TSV GWD Minden",
    "Göppingen": "FRISCH AUF Göppingen",
    "SC DHFK Leipzig": "SC DHfK Leipzig",
}

_team_name_mapping = DEFAULT_TEAM_NAME_MAPPING.copy()


def set_team_name_mapping(mapping: Dict[str, str]) -> None:
    global _team_name_mapping
    _team_name_mapping = mapping.copy()


def normalize_team_name(name: str, mapping: Optional[Dict[str, str]] = None) -> str:
    mapping_to_use = mapping if mapping is not None else _team_name_mapping
    for key, value in mapping_to_use.items():
        if key in name:
            return value
    return name


def find_best_team_match(
    target_name: str, team_list: List[Dict], threshold: float = 0.8
) -> Tuple[Optional[int], Optional[str], float]:
    best_match = None
    best_score = 0.0
    best_id = None
    for team in team_list:
        team_name = team["name"]
        similarity = difflib.SequenceMatcher(
            None, target_name.lower(), team_name.lower()
        ).ratio()
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = team_name
            best_id = team["id"]
    return best_id, best_match, best_score


def extract_team_names_from_match_name(match_name: str) -> Tuple[str, str]:
    if match_name is None or not isinstance(match_name, str) or not match_name.strip():
        raise ValueError("match_name must be a non-empty string")

    if " vs. " in match_name:
        delimiter = " vs. "
    elif " vs " in match_name:
        delimiter = " vs "
    elif " v. " in match_name:
        delimiter = " v. "
    elif " v " in match_name:
        delimiter = " v "
    else:
        raise ValueError(f"Unexpected format in description: {match_name}")

    name_home = match_name.split(delimiter)[0].strip()
    name_away = match_name.split(delimiter)[1].strip()
    return normalize_team_name(name_home), normalize_team_name(name_away)


def _safe_extract_team_names(match_name: object) -> Tuple[Optional[str], Optional[str]]:
    try:
        return extract_team_names_from_match_name(match_name)
    except ValueError:
        return None, None


COLS_TO_DROP = [
    "facility",
    "types",
    "team_id",
    "group_behaviour",
    "track_gps_always",
    "only_imu_data",
    "phases",
    "group_assignment",
    "team_name",
    "track_outside_field",
    "track_while_clock_stopped",
    "track_heart_rate_always",
    "is_gps",
    "sp_usecase",
    "needs_reprocessing",
    "synced",
    "synced_with_sports_cloud",
    "session_id",
    "end_session",
    "duration_sportradar",
    "duration_kinexon",
    "type",
    "timezone_id",
    "group_colors",
    "group_names",
    "organizationId",
    "organization",
    "season",
    "practiceDrillType",
    "internationalReference",
    "status",
    "fixtureNumber",
    "nameLatin",
    "finishRecordingTimeUTC",
    "finishRecordingTimeLocal",
    "startTimeActualUTC",
    "endTimeActualUTC",
    "timesUnconfirmed",
    "locked",
    "placingIfWon",
    "placingIfLost",
    "durationFull",
    "ticketURL",
    "stageCode",
    "stage",
    "seriesCode",
    "poolCode",
    "roundCode",
    "round",
    "liveVideoAvailable",
    "fixtureType",
    "maximumPeriodTypeUsed",
    "competitorType",
    # IMPORTANT: do NOT include "competitors" here
    "venueId",
    "venue",
    "externalId",
    "profileId",
    "includeInStandings",
    "updated",
    "added",
    "estimatedFinishTimeUTC",
    "seriesFixtureNumber",
    "discipline",
    "broadcasts",
    "sellout",
    "liveDataAvailable",
    "team_name_home_kinexon",
    "team_name_away_kinexon",
]


def _safe_next(comps, predicate, value_fn):
    if isinstance(comps, str):
        return None

    if not isinstance(comps, Iterable):
        return None

    for c in comps:
        if isinstance(c, dict) and predicate(c):
            return value_fn(c)

    return None


def _maybe_parse_json_container(value: object) -> object:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{":
        return value

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value

    if isinstance(parsed, (list, dict)):
        return parsed

    return value


def normalize_matches(
    df_fixtures_sportradar_raw: pd.DataFrame,
    df_sessions_kinexon_raw: pd.DataFrame,
    # df_teams_sportradar_raw: pd.DataFrame,
    # df_teams_kinexon_raw: pd.DataFrame,
) -> pd.DataFrame:
    # do not mutate upstream
    k = df_sessions_kinexon_raw.copy()
    s = df_fixtures_sportradar_raw.copy()

    if "season_schedule_item" in k.columns:
        k["season_schedule_item"] = k["season_schedule_item"].map(
            _maybe_parse_json_container
        )

    if "competitors" in s.columns:
        s["competitors"] = s["competitors"].map(_maybe_parse_json_container)

    # fill missing description (map is slightly faster than apply here)
    missing_desc = k["description"].isna()
    if bool(missing_desc.any()):
        k.loc[missing_desc, "description"] = k.loc[
            missing_desc, "season_schedule_item"
        ].map(lambda x: x.get("description") if isinstance(x, dict) else None)

    # extract start_time
    k["start_time"] = k["season_schedule_item"].map(
        lambda x: x.get("start_time") if isinstance(x, dict) else None
    )

    # extract team names (keep your exact semantics)
    k[["team_name_home_kinexon", "team_name_away_kinexon"]] = k["description"].apply(
        lambda x: pd.Series(_safe_extract_team_names(x))
    )
    s[["team_name_home_sportradar", "team_name_away_sportradar"]] = s[
        "nameLocal"
    ].apply(lambda x: pd.Series(_safe_extract_team_names(x)))

    n_missing_kinexon_names = int(k["team_name_home_kinexon"].isna().sum())
    n_missing_sportradar_names = int(s["team_name_home_sportradar"].isna().sum())
    if n_missing_kinexon_names or n_missing_sportradar_names:
        logging.warning(
            "Could not parse team names for %d Kinexon rows and %d Sportradar rows",
            n_missing_kinexon_names,
            n_missing_sportradar_names,
        )

    keys_k = ["team_name_home_kinexon", "team_name_away_kinexon"]
    keys_s = ["team_name_home_sportradar", "team_name_away_sportradar"]

    logging.info(
        "Duplicates: kinexon=%d sportradar=%d",
        int(k.duplicated(subset=keys_k, keep=False).sum()),
        int(s.duplicated(subset=keys_s, keep=False).sum()),
    )

    merged = pd.merge(
        k,
        s,
        how="outer",
        left_on=keys_k,
        right_on=keys_s,
        suffixes=("_kinexon", "_sportradar"),
    )
    logging.info("Number of rows after merge: %d", len(merged))

    # drop noise columns (but never drop competitors before extraction)
    merged = merged.drop(columns=COLS_TO_DROP, errors="ignore")

    # hard fail if competitors is missing here (this is exactly how it becomes "empty again")
    if "competitors" not in merged.columns:
        raise KeyError(
            "Column 'competitors' is missing after merge/drop; entity/score extraction cannot work."
        )

    # keep your markdown safety conversion
    merged = merged.astype("object").where(merged.notna(), None)

    merged["competitors"] = merged["competitors"].map(_maybe_parse_json_container)
    merged["season_schedule_item"] = merged["season_schedule_item"].map(
        _maybe_parse_json_container
    )

    merged["entity_id_home_sportradar"] = merged["competitors"].apply(
        lambda comps: _safe_next(
            comps,
            lambda c: c.get("isHome") is True,
            lambda c: c["entityId"],
        )
    )
    merged["entity_id_away_sportradar"] = merged["competitors"].apply(
        lambda comps: _safe_next(
            comps,
            lambda c: c.get("isHome") is False,
            lambda c: c["entityId"],
        )
    )
    merged["score_home_sportradar"] = merged["competitors"].apply(
        lambda comps: _safe_next(
            comps,
            lambda c: c.get("isHome") is True and "score" in c,
            lambda c: int(c["score"]),
        )
    )
    merged["score_away_sportradar"] = merged["competitors"].apply(
        lambda comps: _safe_next(
            comps,
            lambda c: c.get("isHome") is False and "score" in c,
            lambda c: int(c["score"]),
        )
    )

    merged["team_id_home_kinexon"] = merged["season_schedule_item"].map(
        lambda x: x.get("home_team_id") if isinstance(x, dict) else None
    )
    merged["team_id_away_kinexon"] = merged["season_schedule_item"].map(
        lambda x: x.get("away_team_id") if isinstance(x, dict) else None
    )

    merged = merged.rename(
        columns={
            "startTimeLocal": "start_time_local",
            "startTimeUTC": "start_time_utc",
            "seasonId": "season_id",
            "roundNumber": "round_number",
            "fixture_id": "fixture_id",
            "id": "session_id",
            "team_name_home_sportradar": "team_name_home",
            "team_name_away_sportradar": "team_name_away",
            "entity_id_home_sportradar": "entity_id_home",
            "entity_id_away_sportradar": "entity_id_away",
            "score_home_sportradar": "score_home",
            "score_away_sportradar": "score_away",
            "nameLocal": "match_name",
        }
    )

    desired_order = [
        "season_id",
        "fixture_id",
        "session_id",
        "start_time_local",
        "start_time_utc",
        "start_time",
        "round_number",
        "team_name_home",
        "team_name_away",
        "entity_id_home",
        "entity_id_away",
        "score_home",
        "score_away",
    ]

    merged = merged.drop(
        columns=["season_schedule_item", "competitors"], errors="ignore"
    )
    merged = merged[
        desired_order + [c for c in merged.columns if c not in desired_order]
    ]
    merged = merged.sort_values(by=["round_number", "start_time_local"])

    return merged
