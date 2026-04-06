import difflib

import pandas as pd


PLAYER_OUTPUT_COLUMNS = [
    "fixture_id",
    "entity_id",
    "person_id",
    "name",
    "bib",
    "position",
    "team_name",
    "team_side",
    "date_of_birth",
    "name_family_latin",
    "name_family_local",
    "name_full_latin",
    "name_full_local",
    "name_given_latin",
    "name_given_local",
    "nationality",
    "height",
    "weight",
    "mapped_id",
    "league_id",
    "session_id",
]


def _normalize_match_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _empty_players_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PLAYER_OUTPUT_COLUMNS)


def extract_players_for_match(
    df_match_normalized: pd.DataFrame,
    df_match_events_normalized_setup: pd.DataFrame,
    df_match_detected_shots_normalized: pd.DataFrame,
    df_match_positions_normalized: pd.DataFrame,
    df_match_players_normalized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract players for a match from raw match events and teams data.

    Args:
        match_events_sportradar_raw (pd.DataFrame): DataFrame containing raw match events from Sportradar.
        teams_sportradar_raw (pd.DataFrame): DataFrame containing raw teams data from Sportradar.
    Returns:
        pd.DataFrame: DataFrame containing extracted players for the match.
    """
    df_match_infos = df_match_normalized.copy()  # cols: season_id	fixture_id	session_id	start_time_local	start_time_utc	start_time	round_number	team_name_home	team_name_away	entity_id_home	entity_id_away	score_home	score_away	start_session	description	match_name	attendance	team_id_home_kinexon	team_id_away_kinexon
    df_events = df_match_events_normalized_setup.copy()  # cols: fixture_id	class	entity_id	event_id	event_time	event_type	person_id	bib	name	position	subType	failure_reason	empty_net
    # df_detected_shots = df_match_detected_shots_normalized.copy()  # cols: timestamp	timestamp_ms	game_clock	period	player_id	distance	speed_ball	trajectory	shot_position_x	shot_position_y	hit_position_y	hit_position_z	success	shot_category	goalkeeper_id	shot_type	assisting_player_id	validated	id	event_type	league_id	session_id	fixture_id
    df_positions = df_match_positions_normalized.copy()  # cols: timestamp_ms	formatted_local_time	sensor_id	mapped_id	number	full_name	league_id	group_id	group_name	x_m	y_m	speed_m_s	direction	acceleration	total_distance	metabolic_power	acceleration_load	session_id	fixture_id

    df_match_players = df_match_players_normalized.copy()  # cols: date_of_birth	name_family_latin	name_family_local	name_full_latin	name_full_local	name_given_latin	name_given_local    nationality	person_id	height	weight	fixture_id

    if df_match_infos.empty:
        return _empty_players_frame()

    # Assume df_match_infos is already filtered to this match; if not, we still handle multiple fixtures.
    # Build per-fixture maps.
    home = df_match_infos.loc[
        :, ["fixture_id", "entity_id_home", "team_name_home"]
    ].copy()
    home.columns = ["fixture_id", "entity_id", "team_name"]
    home["team_side"] = "home"

    away = df_match_infos.loc[
        :, ["fixture_id", "entity_id_away", "team_name_away"]
    ].copy()
    away.columns = ["fixture_id", "entity_id", "team_name"]
    away["team_side"] = "away"

    team_lookup = pd.concat([home, away], ignore_index=True)
    team_lookup["fixture_id"] = team_lookup["fixture_id"].astype(str)
    team_lookup["entity_id"] = team_lookup["entity_id"].astype(str)

    if df_events.empty:
        return _empty_players_frame()

    # filter for event_type == "person"
    df_events = df_events[df_events["event_type"] == "person"]

    df_players_in_events = (
        df_events.loc[
            :,
            [
                "fixture_id",
                "entity_id",
                "person_id",
                "name",
                "bib",
                "position",
            ],
        ]
        .dropna(subset=["fixture_id", "entity_id", "person_id"])
        .copy()
    )

    # Enforce: player team must be one of (home, away) for that fixture
    df_players_in_events = df_players_in_events.merge(
        team_lookup,
        on=["fixture_id", "entity_id"],
        how="inner",  # drops players whose entity_id doesn't match home/away for that fixture
    )

    # Enrich with player details from match_players_normalized
    df_players_in_events = df_players_in_events.merge(
        df_match_players,
        on=["fixture_id", "person_id"],
        how="left",
    )

    # insert mapped_id, league_id, session_id from positions where fuzzy match on full_name and group_name
    if not df_positions.empty:
        df_positions_unique_players = df_positions[
            [
                "fixture_id",
                "mapped_id",
                "league_id",
                "session_id",
                "full_name",
                "group_name",
            ]
        ].drop_duplicates()

        def fuzzy_match_player(row):
            name = _normalize_match_text(row.get("name", ""))
            team_name = _normalize_match_text(row.get("team_name", ""))
            candidates = df_positions_unique_players[
                df_positions_unique_players["fixture_id"] == row["fixture_id"]
            ]
            if candidates.empty or not name:
                return pd.Series([None, None, None])

            # Constrain to the player's own team before name matching.
            # Kinexon group_name and Sportradar team_name differ in casing/punctuation,
            # so we use a loose fuzzy match (cutoff=0.4) to resolve the team first.
            if team_name:
                group_names = [
                    group_name
                    for group_name in candidates["group_name"]
                    .map(_normalize_match_text)
                    .unique()
                    .tolist()
                    if group_name
                ]
                team_matches = difflib.get_close_matches(
                    team_name, group_names, n=1, cutoff=0.4
                )
                if team_matches:
                    normalized_group_names = candidates["group_name"].map(
                        _normalize_match_text
                    )
                    candidates = candidates[normalized_group_names == team_matches[0]]

            if candidates.empty:
                return pd.Series([None, None, None])

            # Fuzzy match on full_name within the resolved team
            full_names = [
                full_name
                for full_name in candidates["full_name"]
                .map(_normalize_match_text)
                .tolist()
                if full_name
            ]
            if not full_names:
                return pd.Series([None, None, None])

            name_matches = difflib.get_close_matches(
                name,
                full_names,
                n=1,
                cutoff=0.8,
            )
            if not name_matches:
                return pd.Series([None, None, None])
            matched_name = name_matches[0]
            normalized_full_names = candidates["full_name"].map(_normalize_match_text)
            matched_row = candidates[normalized_full_names == matched_name].iloc[0]
            return pd.Series(
                [
                    matched_row["mapped_id"],
                    matched_row["league_id"],
                    matched_row["session_id"],
                ]
            )

        df_players_in_events[["mapped_id", "league_id", "session_id"]] = (
            df_players_in_events.apply(fuzzy_match_player, axis=1)
        )

    # Unique players per fixture/team
    df_players_in_events = df_players_in_events.drop_duplicates(
        subset=["fixture_id", "entity_id", "person_id"]
    ).reset_index(drop=True)

    df_players_in_events = df_players_in_events.reindex(columns=PLAYER_OUTPUT_COLUMNS)

    return df_players_in_events
