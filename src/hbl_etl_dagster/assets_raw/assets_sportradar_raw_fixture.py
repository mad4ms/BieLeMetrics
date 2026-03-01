import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    Failure,
    MetadataValue,
    asset,
    asset_check,
)

from src.pipelines.raw.sportradar import get_fixture_events as sr_get_fixture_events
from src.pipelines.raw.sportradar import (
    get_players_for_fixture as sr_get_players_for_fixture,
)

fixtures_partition_def = DynamicPartitionsDefinition(name="fixture_partitions")


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_raw",
    compute_kind="duckdb",
    description="Raw Sportradar events for a single fixture (partitioned by fixture_id).",
    partitions_def=fixtures_partition_def,
    deps=[
        "matches_normalized",
    ],
)
def fixture_events_sportradar_raw(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    api = context.resources.sportradar_api
    api.connect()
    fixture_id = context.partition_key

    if not fixture_id:
        raise Failure(description="Missing partition_key (fixture_id).")

    df = sr_get_fixture_events(api=api, fixture_id=str(fixture_id))
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # rename fixtureId column to standardize
    if "fixtureId" in df.columns:
        df = df.rename(columns={"fixtureId": "fixture_id"})

    context.log.info("Fetched %d events (raw) for fixture_id=%s", len(df), fixture_id)
    context.add_output_metadata(
        {
            "fixture_id": str(fixture_id),
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "preview_setup": MetadataValue.md(df.head().to_markdown(index=False)),
            "preview_goals": MetadataValue.md(
                df[df["eventType"] == "goal"].head(100).to_markdown(index=False)
            ),
        }
    )
    if df.empty:
        context.log.warning("No events fetched for fixture_id=%s", fixture_id)
        raise Failure(description=f"No events fetched for fixture_id={fixture_id}")

    nested_cols = []
    for c in df.columns:
        if (
            df[c].dtype == "object"
            and df[c].dropna().map(lambda v: isinstance(v, (dict, list))).any()
        ):
            nested_cols.append(c)

    context.log.info("Nested cols: %s", nested_cols)

    return df


@asset(
    required_resource_keys={"sportradar_api"},
    group_name="sportradar_raw",
    compute_kind="duckdb",
    description="Raw players for a single fixture, derived from that fixture's events (partitioned by fixture_id).",
    partitions_def=fixtures_partition_def,
)
def players_sportradar_raw(
    context: AssetExecutionContext,
    fixture_events_sportradar_raw: pd.DataFrame,
) -> pd.DataFrame:
    api = context.resources.sportradar_api
    fixture_id = context.partition_key

    df_match_events = fixture_events_sportradar_raw[
        fixture_events_sportradar_raw["fixture_id"] == str(fixture_id)
    ]

    if df_match_events is None or df_match_events.empty:
        context.log.warning(
            "No events -> no players (raw) for fixture_id=%s", fixture_id
        )
        return pd.DataFrame()

    df_players = sr_get_players_for_fixture(api=api, df_match_events=df_match_events)
    if not isinstance(df_players, pd.DataFrame):
        df_players = pd.DataFrame(df_players)

    # insert fixture_id column for partitioning
    df_players["fixture_id"] = str(fixture_id)

    context.log.info(
        "Fetched %d players (raw) for fixture_id=%s",
        len(df_players),
        fixture_id,
    )
    context.add_output_metadata(
        {
            "fixture_id": str(fixture_id),
            "n_rows": len(df_players),
            "n_columns": df_players.shape[1],
            "preview": MetadataValue.md(df_players.head().to_markdown(index=False)),
        }
    )
    return df_players


@asset_check(asset=fixture_events_sportradar_raw)
def events_fixture_id_matches_partition_when_present(
    fixture_events_sportradar_raw: pd.DataFrame,
) -> AssetCheckResult:
    # Best-effort check: only enforce if Sportradar includes fixtureId in the payload
    if fixture_events_sportradar_raw.empty:
        return AssetCheckResult(passed=True, metadata={"skipped": "no rows"})

    if "fixtureId" not in fixture_events_sportradar_raw.columns:
        return AssetCheckResult(
            passed=True, metadata={"skipped": "fixtureId column missing"}
        )

    # all non-null values should match (there should typically be exactly one)
    vals = (
        fixture_events_sportradar_raw["fixtureId"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    ok = len(vals) <= 1
    return AssetCheckResult(
        passed=bool(ok),
        metadata={
            "distinct_fixtureId_values": vals[:10],
            "n_distinct": len(vals),
        },
    )


@asset_check(asset=fixture_events_sportradar_raw)
def check_events_have_unique_event_ids_when_present(
    fixture_events_sportradar_raw: pd.DataFrame,
) -> AssetCheckResult:
    if fixture_events_sportradar_raw.empty:
        return AssetCheckResult(passed=True, metadata={"skipped": "no rows"})

    ok = bool(fixture_events_sportradar_raw["eventId"].dropna().astype(str).is_unique)
    n_dup = int(
        fixture_events_sportradar_raw["eventId"].dropna().astype(str).shape[0]
        - fixture_events_sportradar_raw["eventId"].dropna().astype(str).nunique()
    )
    return AssetCheckResult(
        passed=ok,
        metadata={"id_column": "eventId", "n_duplicates": n_dup},
    )


@asset_check(asset=players_sportradar_raw)
def check_players_have_unique_person_ids_when_present(
    players_sportradar_raw: pd.DataFrame,
) -> AssetCheckResult:
    if players_sportradar_raw.empty:
        return AssetCheckResult(passed=True, metadata={"skipped": "no rows"})

    # Only fail when personId AND fixture_id are not unique (if both exist)
    if (
        "personId" in players_sportradar_raw.columns
        and "fixture_id" in players_sportradar_raw.columns
    ):
        df = players_sportradar_raw.dropna(subset=["personId", "fixture_id"])
        n_total = len(df)
        n_unique = df[["personId", "fixture_id"]].astype(str).drop_duplicates().shape[0]
        n_unique_players = df["personId"].dropna().astype(str).nunique()
        ok = n_total == n_unique
        n_dup = n_total - n_unique
        return AssetCheckResult(
            passed=ok,
            metadata={
                "id_columns": ["personId", "fixture_id"],
                "n_duplicates": n_dup,
                "n_rows": n_total,
                "n_unique_players": n_unique_players,
            },
        )


@asset_check(asset=players_sportradar_raw)
def players_fixture_id_matches_events_when_present(
    players_sportradar_raw: pd.DataFrame,
) -> AssetCheckResult:
    if players_sportradar_raw.empty:
        return AssetCheckResult(passed=True, metadata={"skipped": "no rows"})

    if "fixtureId" not in players_sportradar_raw.columns:
        return AssetCheckResult(
            passed=True, metadata={"skipped": "fixtureId column missing"}
        )

    vals = players_sportradar_raw["fixtureId"].dropna().astype(str).unique().tolist()
    ok = len(vals) <= 1
    return AssetCheckResult(
        passed=bool(ok),
        metadata={
            "distinct_fixtureId_values": vals[:10],
            "n_distinct": len(vals),
        },
    )
