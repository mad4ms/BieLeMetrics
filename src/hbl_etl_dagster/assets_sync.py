from dagster import (
    Any,
    Dict,
    List,
    asset,
    AssetExecutionContext,
    MetadataValue,
)
import pandas as pd
from .assets_sportradar_slow import fixtures_partition_def
from .utils.metadata import preview_metadata
from .utils.matching import fuzzy_match_players_to_positions
from .utils.time_sync import sync_goals_with_kinexon
from .utils.sportradar_kinexon_event_mapper import refine_throw_time_for_event


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description=(
        "Backfills players.league_id by matching Sportradar players to Kinexon "
        "positions for the current fixture. Returns a DataFrame of matched players."
    ),
    deps=[
        "kinexon_positions",
        "fixture_events_sportradar",
        "fixture_players_sportradar",
    ],  # ensure kinexon_positions runs first
    metadata={"partition_column": "fixture_id"},
)
def players_merged(
    context: AssetExecutionContext,
    fixture_events_sportradar: pd.DataFrame,
    fixture_players_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Asset wrapping the legacy notebook logic for player ↔ league mapping.
        - Uses the DuckDB IO manager connection.
        - Mutates the `players` table in-place (league_id column).
        - Returns a DataFrame of all players that were updated in this run.
    """
    duckdb_io_manager = context.resources.io_manager
    fixture_id = context.partition_key

    # Filter fixture_events_match for current partition
    fixture_events_sportradar["fixture_id"] = fixture_events_sportradar[
        "fixture_id"
    ].astype(str)
    fixture_events_sportradar = fixture_events_sportradar[
        fixture_events_sportradar["fixture_id"] == fixture_id
    ]

    # Filter fixture_players for current partition
    fixture_players_sportradar["fixture_id"] = fixture_players_sportradar[
        "fixture_id"
    ].astype(str)
    fixture_players_sportradar = fixture_players_sportradar[
        fixture_players_sportradar["fixture_id"] == fixture_id
    ]

    # there should be only one session_id in this fixture, get it
    session_ids = fixture_events_sportradar["session_id"].dropna().unique()
    assert (
        len(session_ids) == 1
    ), f"Multiple or no session_ids found for fixture {fixture_id}"
    session_id = session_ids[0]

    # fixture_events is now partitioned, so it contains only events for this fixture
    if fixture_events_sportradar.empty:
        context.log.warning(
            f"No events for fixture {fixture_id}. Skipping player merge."
        )
        return pd.DataFrame()

    df_players_fixture = fixture_players_sportradar.copy()

    with duckdb_io_manager._conn() as con:
        # Process the single fixture
        df_events_fixture = fixture_events_sportradar
        person_ids = df_events_fixture["person_id"].dropna().unique().tolist()
        person_ids_fixture_player = (
            df_players_fixture["person_id"].dropna().unique().tolist()
        )
        # assert that person_ids_fixture_player is in person_ids
        missing_person_ids = set(person_ids_fixture_player) - set(person_ids)
        assert (
            not missing_person_ids
        ), f"Missing person IDs in fixture_events_sportradar: {missing_person_ids}"

        # Fetch unique player info from Kinexon positions for this session
        try:
            df_kin_player_in_positions = con.execute(
                f"""
                SELECT DISTINCT
                    "full name"  AS full_name_kinexon,
                    "group name" AS group_name_kinexon,
                    "league id"  AS league_id,
                    "mapped id"  AS mapped_id
                FROM kinexon_positions
                WHERE "full name" IS NOT NULL
                AND "league id" IS NOT NULL
                AND "group name" IS NOT NULL
                AND session_id = {session_id}
                """
            ).df()
        except Exception as e:
            context.log.warning(
                f"Fixture {fixture_id}: Could not query positions for session {session_id}. Error: {e}"
            )
            return pd.DataFrame()

        if df_kin_player_in_positions.empty:
            context.log.info(
                f"Fixture {fixture_id}: No Kinexon positions found for session {session_id}."
            )
            return pd.DataFrame()

        # remove where "ball" or "Ball" is in league_id
        df_kin_player_in_positions = df_kin_player_in_positions[
            ~df_kin_player_in_positions["league_id"].str.contains(
                "ball", case=False, na=False
            )
        ]

        context.log.info(
            f"Fixture {fixture_id}: Performing fuzzy matching of players to Kinexon positions."
            f"Unique team names in Kinexon positions: {df_kin_player_in_positions['group_name_kinexon'].unique()}"
            f"Unique team names in fixture players: {df_players_fixture['team_name'].unique()}"
        )

        df_players_matched = fuzzy_match_players_to_positions(
            df_players=df_players_fixture,
            df_positions=df_kin_player_in_positions,
            score_threshold=80,
            logger=context.log,
        )

        fixture_id = context.partition_key
        df_players_matched["fixture_id"] = str(fixture_id)

    if df_players_matched.empty:
        return pd.DataFrame()

    context.log.info(
        f"Total unique players updated with league_id: {len(df_players_matched)}"
    )
    context.add_output_metadata(
        {
            "total_unique_players_updated_with_league_id": len(
                df_players_matched
            ),
            "n_players_in_raw_kinexon_positions": len(
                df_kin_player_in_positions
            ),
            "n_players_in_fixture_events": len(df_players_fixture),
        }
    )
    context.add_output_metadata(preview_metadata(df_players_matched))
    return df_players_matched


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description="Sportradar goal events synchronized with the nearest Kinexon event for the current fixture.",
)
def sportradar_goals_synced(
    context: AssetExecutionContext,
    fixtures_sportradar: pd.DataFrame,
    fixture_events_sportradar: pd.DataFrame,
    kinexon_events: pd.DataFrame,
    players_merged: pd.DataFrame,
) -> pd.DataFrame:
    """Match Sportradar goals to Kinexon timeline for a single fixture partition."""

    fixture_id = context.partition_key
    fixture_events_sportradar["fixture_id"] = fixture_events_sportradar[
        "fixture_id"
    ].astype(str)
    fixture_events_sportradar = fixture_events_sportradar[
        fixture_events_sportradar["fixture_id"] == fixture_id
    ]
    kinexon_events["fixture_id"] = kinexon_events["fixture_id"].astype(str)
    kinexon_events = kinexon_events[kinexon_events["fixture_id"] == fixture_id]
    players_merged["fixture_id"] = players_merged["fixture_id"].astype(str)
    players_merged = players_merged[players_merged["fixture_id"] == fixture_id]
    fixtures_sportradar["fixture_id"] = fixtures_sportradar[
        "fixture_id"
    ].astype(str)
    fixture_info = fixtures_sportradar[
        fixtures_sportradar["fixture_id"] == fixture_id
    ]

    context.log.info(
        "Starting goal sync for fixture %s. Sportradar events: %d, Kinexon events: %d, Players merged: %d",
        fixture_id,
        len(fixture_events_sportradar),
        len(kinexon_events),
        len(players_merged),
    )

    if fixture_events_sportradar.empty or kinexon_events.empty:
        context.log.warning(
            f"One or both input DataFrames are empty for fixture {fixture_id}."
        )
        return pd.DataFrame()

    # start_time is in fixture_events_match where event_type == 'fixture' and sub_type == 'start'
    start_time = (
        fixture_events_sportradar[
            (fixture_events_sportradar["event_type"] == "fixture")
            & (fixture_events_sportradar["sub_type"] == "start")
        ]["event_time"].iloc[0]
        if not fixture_events_sportradar.empty
        else None
    )
    start_time_kinexon = (
        pd.to_datetime(
            kinexon_events["timestamp_ms"].min(), unit="ms", utc=True
        )
        if not kinexon_events.empty
        else None
    )
    context.log.info(
        f"Fixture {fixture_id} start time: {start_time}"
        f", Kinexon start time: {start_time_kinexon}"
    )
    df_goals = fixture_events_sportradar[
        fixture_events_sportradar["event_type"] == "goal"
    ].copy()
    if df_goals.empty:
        context.log.warning(
            f"No goal events left after filtering fixture {fixture_id}."
        )
        return pd.DataFrame()

    df_goals["event_time"] = pd.to_datetime(
        df_goals["event_time"], utc=True, errors="coerce"
    )
    df_goals["event_time_ms"] = (
        df_goals["event_time"].astype("int64") // 1_000_000
    )
    df_goals = df_goals.dropna(subset=["event_time_ms", "fixture_id"])
    df_goals["fixture_id"] = df_goals["fixture_id"].astype(str)

    if players_merged.empty or "kin_league_id" not in players_merged.columns:
        context.log.warning(
            f"No matched players for fixture {fixture_id}. Goalkeeper IDs may stay null."
        )
        player_league_map = {}
    else:
        context.log.debug(f"players_merged shape: {players_merged.shape}")
        context.log.debug(
            f"players_merged columns: {players_merged.columns.tolist()}"
        )
        context.log.debug(
            f"Non-null 'kin_league_id' count: {players_merged['kin_league_id'].notna().sum()}"
        )
        context.log.debug(
            f"Unique 'person_id' count: {players_merged['person_id'].nunique()}"
        )
        # unique team names
        context.log.debug(
            f"Unique team names: {players_merged['team_name'].unique().tolist()}"
        )
        # unique team names in df_goals
        context.log.debug(
            f"Unique team names in df_goals: {df_goals['team_name'].unique().tolist()}"
        )

        player_league_map = (
            players_merged[["person_id", "kin_league_id"]]
            .dropna(subset=["person_id", "kin_league_id"])
            .drop_duplicates()
            .set_index("person_id")["kin_league_id"]
            .to_dict()
        )
    context.log.info(
        f"Found {len(player_league_map)} players with mapped league IDs for fixture {fixture_id}."
    )

    context.log.info(
        f"Mapping league IDs for goalkeepers and shooters for fixture {fixture_id}."
    )
    df_goals["goalkeeper_league_id"] = df_goals["goalkeeper_id"].map(
        player_league_map
    )

    df_goals["person_league_id"] = df_goals["person_id"].map(player_league_map)

    context.log.info(
        f"Mapped league IDs for {df_goals['goalkeeper_league_id'].notna().sum()} goalkeepers"
        f" and {df_goals['person_league_id'].notna().sum()} shooters for fixture {fixture_id}."
    )

    df_kinexon_events = kinexon_events.copy()
    # fixture_id_map = (
    #     df_goals[["fixture_id", "session_id"]]
    #     .drop_duplicates()
    #     .set_index("session_id")["fixture_id"]
    #     .to_dict()
    # )
    # df_kinexon_events["fixture_id"] = df_kinexon_events["session_id"].map(
    #     fixture_id_map
    # )
    context.log.info(
        f"Mapped fixture IDs for Kinexon events for fixture {fixture_id}."
        f" Unique fixture ids in input: {df_kinexon_events['fixture_id'].unique().tolist()}"
    )
    df_kinexon_events["fixture_id"].fillna(fixture_id, inplace=True)
    df_kinexon_events["timestamp_ms"] = pd.to_numeric(
        df_kinexon_events["timestamp_ms"], errors="coerce"
    )
    df_kinexon_events = df_kinexon_events.dropna(subset=["timestamp_ms"])
    context.log.info(
        f"Dropped Kinexon events with invalid timestamps for fixture {fixture_id}."
        f" Remaining events: {len(df_kinexon_events)}"
    )
    df_kinexon_events["fixture_id"] = df_kinexon_events["fixture_id"].astype(
        str
    )

    tolerance_ms = (context.op_config or {}).get("tolerance_ms", 30_000)

    df_final_synced = sync_goals_with_kinexon(
        df_events=df_goals,
        df_kinexon=df_kinexon_events,
        tolerance_ms=tolerance_ms,
    )

    total_goals = len(df_goals)
    matched_goals = len(df_final_synced)
    match_rate = (matched_goals / total_goals * 100) if total_goals > 0 else 0

    date_game = df_goals["event_time"].iloc[0]
    # convert first ts in ms in kinexon to date
    date_game_kinexon = (
        pd.to_datetime(df_kinexon_events["timestamp_ms"].iloc[0], unit="ms")
        if not df_kinexon_events.empty
        else "unknown"
    )
    competitors = "_vs_".join(df_goals["team_name"].unique())
    round_number = (
        fixture_info["round_number"].iloc[0]
        if "round_number" in fixture_info.columns
        else "unknown"
    )

    context.log.info(
        "Fixture %s goal sync complete. Matched %d/%d goals (%.2f%%). Date: %s, Date Kinexon: %s, Teams: %s, Round: %s",
        fixture_id,
        matched_goals,
        total_goals,
        match_rate,
        date_game,
        date_game_kinexon,
        competitors,
        round_number,
    )

    context.add_output_metadata(
        {
            "fixture_id": fixture_id,
            "n_rows": matched_goals,
            "n_columns": df_final_synced.shape[1],
            "total_goals_processed": total_goals,
            "match_rate": f"{match_rate:.2f}%",
            "time_tolerance_ms": tolerance_ms,
            "date_game": str(date_game),
            "date_game_kinexon": str(date_game_kinexon),
            "teams": competitors,
            "round_number": round_number,
            "preview": (
                MetadataValue.md(
                    df_final_synced.head(10).to_markdown(index=False)
                )
                if not df_final_synced.empty
                else MetadataValue.md("*(empty)*")
            ),
        }
    )

    return df_final_synced


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description="Refines throw timestamps for synced Sportradar goals using Kinexon position data for the current fixture.",
    deps=["kinexon_positions"],
    metadata={"partition_column": "fixture_id"},
)
def sportradar_goals_refined(
    context: AssetExecutionContext,
    sportradar_goals_synced: pd.DataFrame,
) -> pd.DataFrame:
    """
    Iterates over synced goals, fetches Kinexon positions per fixture,
    and applies the throw-time refinement heuristic.
    """
    duckdb_io_manager = context.resources.io_manager
    fixture_id = context.partition_key
    df = sportradar_goals_synced.copy()
    df["fixture_id"] = df["fixture_id"].astype(str)
    df = df[df["fixture_id"] == fixture_id]

    if df.empty:
        context.log.warning(
            f"No synced goals for fixture {fixture_id} in this partition."
            f" For unique fixture ids in {sportradar_goals_synced['fixture_id'].unique().tolist()}"
        )
        return df.head(0)

    context.log.info(
        f"Refining {len(df)} synced goals for fixture {fixture_id}."
        f" Unique fixture ids in input: {df['fixture_id'].unique().tolist()}"
    )

    # We will collect the results here
    refined_rows = []

    # Since we are partitioned by fixture, we expect only one fixture_id in the dataframe
    # But we can still group by just to be safe and reuse logic
    for f_id, df_fixture_goals in df.groupby("fixture_id"):
        if str(f_id) != str(fixture_id):
            context.log.warning(
                f"Found data for fixture {f_id} in partition {fixture_id}. Processing anyway."
            )

        date_game = df_fixture_goals["event_time"].iloc[0]
        date_only = date_game.date() if pd.notna(date_game) else "unknown"
        competitors = "_vs_".join(df_fixture_goals["team_name"].unique())
        context.log.info(
            f"Refining {len(df_fixture_goals)} goals for fixture {f_id} ({competitors}) on {date_only}"
        )

        # Fetch positions for this fixture
        try:
            with duckdb_io_manager._conn() as con:
                df_positions = con.execute(
                    f"SELECT * FROM kinexon_positions WHERE fixtureId = '{f_id}'"
                ).df()
        except Exception as e:
            context.log.error(
                f"Error fetching positions for fixture {f_id}: {e}"
            )
            df_positions = pd.DataFrame()

        if df_positions.empty:
            context.log.warning(
                f"No positions found for fixture {f_id}. Skipping refinement."
            )
            # Append original rows without refinement
            for _, row in df_fixture_goals.iterrows():
                refined_rows.append(row)
            continue

        # Pre-process positions
        if "ts" not in df_positions.columns:
            if "ts in ms" in df_positions.columns:
                df_positions["ts"] = pd.to_datetime(
                    df_positions["ts in ms"],
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

        # Build fixture_to_session map
        fixture_to_session = {}
        if "session_id" in df_positions.columns:
            s_ids = df_positions["session_id"].dropna().unique()
            if len(s_ids) > 0:
                fixture_to_session[f_id] = s_ids[0]
        else:
            context.log.warning(
                f"No session_id column in positions for fixture {f_id}."
            )

        teams = df_fixture_goals["team_name"].unique().tolist()

        # store goal_pos per team per period
        goal_pos_map = {team: {1: None, 2: None} for team in teams}

        for team in teams:

            # --- goalkeeper ids for this team ---
            goalkeeper_league_ids = (
                df_fixture_goals.loc[
                    df_fixture_goals["team_name"] == team,
                    "goalkeeper_league_id",
                ]
                .dropna()
                .unique()
            )

            # --- period 1 events for this team ---
            df_events_p1_gk = df_fixture_goals[
                (df_fixture_goals["period_id"] == 1)
                & (df_fixture_goals["team_name"] == team)
            ]

            if df_events_p1_gk.empty or len(goalkeeper_league_ids) == 0:
                last_event_p1 = None
            else:
                last_event_p1 = df_events_p1_gk.sort_values(
                    "event_time_ms"
                ).iloc[-1]

            if last_event_p1 is not None:
                split_ts_ms = int(last_event_p1["event_time_ms"])
            else:
                split_ts_ms = None

            # --- build df_pos_period1 / df_pos_period2 ---
            if split_ts_ms is not None:
                df_pos_period1 = df_positions[
                    df_positions["ts in ms"] <= split_ts_ms
                ]
                df_pos_period2 = df_positions[
                    df_positions["ts in ms"] > split_ts_ms
                ]
            else:
                df_pos_period1 = df_positions.head(0)
                df_pos_period2 = df_positions.head(0)

            # --- goalkeeper positions period 1 ---
            df_gk = df_pos_period1[
                df_pos_period1["league id"].isin(goalkeeper_league_ids)
            ]
            if not df_gk.empty:
                median_x = df_gk["x in m"].median()
                goal_pos_map[team][1] = 0 if median_x < 20 else 40

            # --- goalkeeper positions period 2 ---
            df_gk = df_pos_period2[
                df_pos_period2["league id"].isin(goalkeeper_league_ids)
            ]
            if not df_gk.empty:
                median_x = df_gk["x in m"].median()
                goal_pos_map[team][2] = 0 if median_x < 20 else 40

        # --- assign into df_fixture_goals ---
        df_fixture_goals["goal_position"] = df_fixture_goals.apply(
            lambda row: goal_pos_map.get(row["team_name"], {}).get(
                row["period_id"], None
            ),
            axis=1,
        )

        # Apply refinement
        for _, row in df_fixture_goals.iterrows():
            try:
                diag_series = refine_throw_time_for_event(
                    row=row,
                    df_positions_all=df_positions,
                    fixture_to_session=fixture_to_session,
                    plot=False,
                )
                # Merge diag_series into row
                combined = pd.concat([row, diag_series])
                # Remove duplicate index labels (keep last/refined)
                combined = combined.loc[
                    ~combined.index.duplicated(keep="last")
                ]
                refined_rows.append(combined)
            except Exception as e:
                context.log.error(
                    f"Error refining event {row.get('eventId')}: {e}"
                )
                refined_rows.append(row)

    if not refined_rows:
        return pd.DataFrame()

    df_refined = pd.DataFrame(refined_rows)

    context.add_output_metadata(preview_metadata(df_refined))

    return df_refined


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager"},
    group_name="synced_data",
    compute_kind="duckdb",
    description="Refines throw timestamps for synced Sportradar goals using Kinexon position data for the current fixture.",
    deps=["kinexon_positions"],
    metadata={"partition_column": "fixture_id"},
)
def positions_for_throw_time(
    context: AssetExecutionContext,
    sportradar_goals_refined: pd.DataFrame,
    fixtures_sportradar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter kinexon_positions by throw timestamps for this fixture,
    attach eventId, and produce detailed metadata.
    """
    duckdb_io_manager = context.resources.io_manager
    fixture_id = context.partition_key

    # Filter goals for this fixture
    df_goals = sportradar_goals_refined.copy()
    df_goals["fixture_id"] = df_goals["fixture_id"].astype(str)
    df_goals = df_goals[df_goals["fixture_id"] == fixture_id]

    df_fixture_info = fixtures_sportradar.copy()
    df_fixture_info["fixture_id"] = df_fixture_info["fixture_id"].astype(str)
    df_fixture_info = df_fixture_info[
        df_fixture_info["fixture_id"] == fixture_id
    ]

    # Create timestamp-to-eventId mapping
    df_map = (
        df_goals[["refined_throw_ts_ms", "event_id"]]
        .dropna(subset=["refined_throw_ts_ms"])
        .drop_duplicates(subset=["refined_throw_ts_ms"])
        .rename(columns={"refined_throw_ts_ms": "ts in ms"})
    )

    list_throw_time_in_fixture = df_map["ts in ms"].unique().tolist()

    context.log.info(
        f"Unique refined throw times in fixture {fixture_id}: {list_throw_time_in_fixture}"
    )

    if not list_throw_time_in_fixture:
        context.log.warning(
            f"No throw timestamps found for fixture {fixture_id}. Returning empty frame."
        )
        return sportradar_goals_refined.head(0)

    # SQL list
    placeholders = ", ".join([str(int(x)) for x in list_throw_time_in_fixture])

    # Query Kinexon
    with duckdb_io_manager._conn() as con:
        df_pos = con.execute(
            f"""
            SELECT *
            FROM kinexon_positions
            WHERE fixtureId = '{fixture_id}'
              AND "ts in ms" IN ({placeholders})
            """
        ).df()

    if df_pos.empty:
        context.log.warning(
            f"No Kinexon positions for fixture {fixture_id}. "
            f"Available fixture_ids in sportradar: {sportradar_goals_refined['fixture_id'].unique().tolist()}"
        )
        return df_pos.head(0)

    context.log.info(
        f"Returning {len(df_pos)} Kinexon positions for fixture {fixture_id} and home team {df_fixture_info['name_team_home'].iloc[0]}."
    )

    # flensburg fix: y-axis has 12.5m offset
    if (
        not df_fixture_info.empty
        and "name_team_home" in df_fixture_info.columns
        and df_fixture_info["name_team_home"].iloc[0]
        == "SG Flensburg-Handewitt"
    ):
        context.log.info(
            f"Applying y-axis offset fix for fixture {fixture_id} (SG Flensburg-Handewitt home)."
        )
        df_pos["y in m"] = df_pos["y in m"] - 12.5

    # Merge eventId
    df_merged = df_pos.merge(df_map, on="ts in ms", how="left")
    # insert fixture_id column
    df_merged["fixture_id"] = fixture_id

    # Compute merge statistics
    unique_kin_timestamps = df_pos["ts in ms"].nunique()
    unique_goal_timestamps = len(list_throw_time_in_fixture)

    merged_with_event = df_merged["event_id"].notna().sum()
    missing_event = df_merged["event_id"].isna().sum()

    coverage_ratio = (
        merged_with_event / len(df_merged) if len(df_merged) > 0 else 0.0
    )

    # Additional logging
    context.log.info(
        f"Kinexon timestamps found: {unique_kin_timestamps}, "
        f"Sportradar throw timestamps: {unique_goal_timestamps}, "
        f"Merged rows with eventId: {merged_with_event}, "
        f"Missing eventId: {missing_event}, "
        f"Coverage: {coverage_ratio:.2%}"
    )

    # Build Dagster metadata dict
    metadata = preview_metadata(df_merged)
    metadata.update(
        {
            "kinexon_rows": len(df_pos),
            "unique_kinexon_timestamps": unique_kin_timestamps,
            "unique_throw_timestamps_in_sportradar": unique_goal_timestamps,
            "merged_rows_with_eventId": int(merged_with_event),
            "rows_missing_eventId": int(missing_event),
            "coverage_ratio": float(coverage_ratio),
        }
    )

    context.add_output_metadata(metadata)
    return df_merged


from .utils.goal_rendering import render_goal_with_multifreeze, OUT_DIR


@asset(
    partitions_def=fixtures_partition_def,
    required_resource_keys={"io_manager"},
    group_name="renders",
    compute_kind="python",
    description=(
        "Render MP4 clips for the first five refined throw events of a fixture "
        "using Kinexon positions and Sportradar metadata."
    ),
    deps=["positions_for_throw_time"],
    metadata={"partition_column": "fixture_id"},
)
def rendered_throw_videos_first5(
    context: AssetExecutionContext,
    fixtures_sportradar: pd.DataFrame,
    sportradar_goals_refined: pd.DataFrame,
    positions_for_throw_time: pd.DataFrame,
) -> pd.DataFrame:
    """
    For the current fixture partition:

    1. Take refined goals that actually have a refined_throw_ts and a matching
       entry in positions_for_throw_time.
    2. Select the first five (by event_time).
    3. Load kinexon_positions for this fixture from DuckDB.
    4. For each goal, build freeze markers and call render_goal_with_multifreeze.
    5. Return a small DataFrame listing rendered paths.
    """
    fixture_id = context.partition_key

    # --- filter refined goals for this fixture ---
    df_goals = sportradar_goals_refined.copy()
    df_goals["fixture_id"] = df_goals["fixture_id"].astype(str)
    df_goals = df_goals[df_goals["fixture_id"] == fixture_id]

    if df_goals.empty:
        context.log.warning(
            f"No refined goals for fixture {fixture_id}. Nothing to render."
        )
        return df_goals.head(0)

    df_fixture_info = fixtures_sportradar.copy()
    df_fixture_info["fixture_id"] = df_fixture_info["fixture_id"].astype(str)
    df_fixture_info = df_fixture_info[
        df_fixture_info["fixture_id"] == fixture_id
    ]

    # --- restrict to events that actually have positions_for_throw_time rows ---
    df_pos_throw = positions_for_throw_time.copy()
    if "fixture_id" in df_pos_throw.columns:
        df_pos_throw["fixture_id"] = df_pos_throw["fixture_id"].astype(str)
        df_pos_throw = df_pos_throw[df_pos_throw["fixture_id"] == fixture_id]

    if df_pos_throw.empty:
        context.log.warning(
            f"No positions_for_throw_time rows for fixture {fixture_id}. "
            "Skipping rendering."
        )
        return df_goals.head(0)

    if "event_id" in df_pos_throw.columns:
        valid_event_ids = (
            df_pos_throw["event_id"].dropna().drop_duplicates().tolist()
        )
        df_goals = df_goals[df_goals["event_id"].isin(valid_event_ids)]

    # require refined_throw_ts
    if "refined_throw_ts" in df_goals.columns:
        df_goals = df_goals[df_goals["refined_throw_ts"].notna()]

    if df_goals.empty:
        context.log.warning(
            f"No refined goals with usable throw timestamps for fixture {fixture_id}."
        )
        return df_goals.head(0)

    # --- pick first five events (sorted by event_time, fallback to refined_throw_ts) ---
    sort_cols = [
        c for c in ["event_time", "refined_throw_ts"] if c in df_goals.columns
    ]
    if sort_cols:
        df_goals = df_goals.sort_values(sort_cols)
    df_goals = df_goals.head(5)

    context.log.info(
        f"Rendering {len(df_goals)} events for fixture {fixture_id}."
    )

    # --- load kinexon_positions once for this fixture ---
    duckdb_io_manager = context.resources.io_manager
    with duckdb_io_manager._conn() as con:
        df_positions = con.execute(
            f"SELECT * FROM kinexon_positions WHERE fixtureId = '{fixture_id}'"
        ).df()

    if df_positions.empty:
        context.log.warning(
            f"No kinexon_positions for fixture {fixture_id}. Cannot render clips."
        )
        return df_goals.head(0)

    context.log.info(
        f"Home team for fixture {fixture_id}: {df_fixture_info['name_team_home'].iloc[0]}"
    )

    # flensburg fix: y-axis has 12.5m offset
    if (
        not df_fixture_info.empty
        and "name_team_home" in df_fixture_info.columns
        and df_fixture_info["name_team_home"].iloc[0]
        == "SG Flensburg-Handewitt"
    ):
        context.log.info(
            f"Applying y-axis offset fix for fixture {fixture_id} (SG Flensburg-Handewitt home)."
        )
        df_positions["y in m"] = df_positions["y in m"] - 12.5

    rendered_records: List[Dict[str, Any]] = []

    for _, row in df_goals.iterrows():
        event_id = row.get("event_id")
        context.log.info(
            f"Rendering event_id={event_id} for fixture {fixture_id}."
        )

        freeze_markers: List[Dict[str, Any]] = []

        # Sportradar event time marker
        if "event_time" in row and pd.notna(row["event_time"]):
            freeze_markers.append(
                {
                    "name": "SR event_time",
                    "ts": row["event_time"],
                    "color": (0, 0, 255),  # red frame
                    "seconds": 0.5,
                }
            )

        # Kinexon original sync timestamp, if present
        if "kin_timestamp" in row and pd.notna(row.get("kin_timestamp")):
            freeze_markers.append(
                {
                    "name": "Kinexon sync",
                    "ts": row["kin_timestamp"],
                    "color": (0, 255, 255),  # yellow frame
                    "seconds": 0.5,
                }
            )

        # Refined throw time marker (primary)
        if "refined_throw_ts" in row and pd.notna(row.get("refined_throw_ts")):
            freeze_markers.append(
                {
                    "name": "Refined throw",
                    "ts": row["refined_throw_ts"],
                    "color": (0, 255, 0),  # green frame
                    "seconds": 1.0,
                }
            )

        if not freeze_markers:
            context.log.warning(
                f"Event {event_id}: no valid freeze markers, skipping."
            )
            continue

        out_path = render_goal_with_multifreeze(
            df_positions=df_positions,
            row_goal=row,
            freeze_markers=freeze_markers,
        )

        if out_path is None:
            context.log.warning(
                f"Rendering failed or skipped for event {event_id} in fixture {fixture_id}."
            )
            continue

        rendered_records.append(
            {
                "fixture_id": fixture_id,
                "event_id": event_id,
                "video_path": str(out_path),
                "preview_image_path": str(out_path.with_suffix(".png")),
            }
        )

    if not rendered_records:
        context.log.warning(
            f"No renders were produced for fixture {fixture_id}."
        )
        return df_goals.head(0)

    df_rendered = pd.DataFrame(rendered_records)

    context.add_output_metadata(preview_metadata(df_rendered))
    context.add_output_metadata(
        {
            "n_events_rendered": len(df_rendered),
            "output_dir": str(OUT_DIR),
            "events": ", ".join(
                str(e) for e in df_rendered["event_id"].tolist()
            ),
        }
    )

    return df_rendered
