import logging
from typing import Dict, Optional, Tuple, List

import pandas as pd


def normalize_match_players(
    df_players_sportradar_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize match players from raw players data.

    Args:
        df_players_raw (pd.DataFrame): DataFrame containing raw players data.
    Returns:
        pd.DataFrame: Normalized match players DataFrame.
    """
    df_players = df_players_sportradar_raw.copy()

    # cols: added	deceased	dob	externalId	gender	historicalNames	images	languageLocal	nameAbbreviated	nameFamilyLatin	nameFamilyLocal	nameFullLatin	nameFullLocal	nameGivenLatin	nameGivenLocal	nationality	organizationId	personId	representing	status	updated	additionalDetails_height	additionalDetails_weight	organization_id	organization_resourceType	fixture_id

    dict_cols_to_keep = {
        # "added": "added",
        # "deceased": "deceased",
        "dob": "date_of_birth",
        # "externalId": "external_id",
        # "gender": "gender",
        # "historicalNames": "historical_names",
        # "images": "images",
        # "languageLocal": "language_local",
        # "nameAbbreviated": "name_abbreviated",
        "nameFamilyLatin": "name_family_latin",
        "nameFamilyLocal": "name_family_local",
        "nameFullLatin": "name_full_latin",
        "nameFullLocal": "name_full_local",
        "nameGivenLatin": "name_given_latin",
        "nameGivenLocal": "name_given_local",
        "nationality": "nationality",
        # "organizationId": "organization_id",
        "personId": "person_id",
        # "representing": "representing",
        # "status": "status",
        # "updated": "updated",
        "additionalDetails_height": "height",
        "additionalDetails_weight": "weight",
        # "organization_resourceType": "organization_resource_type",
        "fixture_id": "fixture_id",
    }
    df_players = df_players.rename(columns=dict_cols_to_keep)

    # only keep relevant columns
    df_players = df_players[list(dict_cols_to_keep.values())]

    logging.info("Normalized %d players", len(df_players))

    return df_players
