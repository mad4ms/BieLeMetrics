import pandas as pd
from thefuzz import process
import logging


def fuzzy_match_players_to_positions(
    df_players: pd.DataFrame,
    df_positions: pd.DataFrame,
    score_threshold: int = 80,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Return subset of players with added kinexon league_id column.
    Matches players to Kinexon positions using fuzzy string matching on names.
    """
    matched_rows = []

    # Ensure we have unique names from positions to match against
    position_names = df_positions["full_name_kinexon"].unique().tolist()

    if logger:
        logger.debug(
            f"Starting fuzzy matching for {len(df_players)} players against {len(position_names)} Kinexon positions."
        )

    position_groups = df_positions["group_name_kinexon"].unique().tolist()

    for _, player_row in df_players.iterrows():
        player_name = player_row["name_full_local"]
        player_team = player_row["team_name"]

        # Extract best match
        match_result = process.extractOne(player_name, position_names)
        if not match_result:
            continue

        match, score = match_result

        if score >= score_threshold:
            # Get the league_id for the matched name
            # Assuming names are unique enough within a session/fixture context
            kinexon_row = df_positions[
                df_positions["full_name_kinexon"] == match
            ].iloc[0]

            player_row["kin_league_id"] = kinexon_row["league_id"]
            player_row["kin_mapped_id"] = kinexon_row["mapped_id"]
            matched_rows.append(player_row)
        else:
            if logger:
                logger.debug(
                    f"Player '{player_name}' did not find a good match (score: {score}, {match})."
                )

    if logger:
        logger.info(
            f"Matched {len(matched_rows)} out of {len(df_players)} players."
        )

    if not matched_rows:
        return pd.DataFrame()

    return pd.DataFrame(matched_rows)
