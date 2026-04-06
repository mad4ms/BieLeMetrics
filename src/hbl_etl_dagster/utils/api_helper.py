"""Configuration and shared helpers for HBL ETL scripts.

This module extracts the top-level constants and creates helpers to
get a DuckDB connection and initialized API client. It's a direct
translation of the notebook's configuration cells.
"""

import datetime
import os
from typing import Optional
from urllib.parse import urlparse

import duckdb
from dotenv import load_dotenv
from sportradar_datacore_api.handball import HandballAPI

load_dotenv()

NAME_COMPETITION = "1. Handball-Bundesliga"
NAME_SEASON = "DAIKIN HBL 2024/25"
YEAR_SEASON = int(NAME_SEASON.split()[-1].split("/")[0])
YEARS_SEASON = NAME_SEASON.split()[-1].replace("/", "-")

PATH_TO_OUTPUT = os.path.join(os.getcwd(), "..", "data", f"season_{YEARS_SEASON}")
os.makedirs(PATH_TO_OUTPUT, exist_ok=True)

date = datetime.date.today().strftime("%Y-%m-%d")


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _validate_http_url(name: str, value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Environment variable {name} must be a valid absolute http(s) URL, got: {value!r}"
        )
    return value


def _get_required_http_url(name: str) -> str:
    return _validate_http_url(name, _get_required_env(name))


def get_duckdb_connection(db_path: Optional[str] = None):
    """Return a DuckDB connection using the same relative path as the notebook.

    By default it uses ../data/mydb{YEARS_SEASON}.duckdb to match the notebook.
    """
    if db_path is None:
        db_path = f"./data/mydb_{date}_{YEARS_SEASON}.duckdb"
    return duckdb.connect(db_path)


def get_api_sportradar():
    """Initialize and return the HandballAPI client using environment variables.

    Matches the notebook's HandballAPI initialisation.
    """
    api = HandballAPI(
        base_url=_get_required_http_url("BASE_URL"),
        auth_url=_get_required_http_url("AUTH_URL"),
        client_id=_get_required_env("CLIENT_ID"),
        client_secret=_get_required_env("CLIENT_SECRET"),
        org_id=_get_required_env("CLIENT_ORGANIZATION_ID"),
        scopes=["read:organization"],
        sport="handball",
    )
    return api


def get_api_kinexon():
    """Initialize and return the Kinexon API client using environment variables.

    Matches the notebook's KinexonAPI initialisation.
    """
    from kinexon_handball_api.handball import HandballAPI

    endpoint_session = _get_required_http_url("ENDPOINT_KINEXON_SESSION")
    endpoint_main = _get_required_http_url("ENDPOINT_KINEXON_MAIN")
    endpoint_api_raw = os.getenv("ENDPOINT_KINEXON_API")
    endpoint_api = (
        _validate_http_url("ENDPOINT_KINEXON_API", endpoint_api_raw.strip())
        if endpoint_api_raw and endpoint_api_raw.strip()
        else endpoint_main
    )

    api = HandballAPI(
        base_url=endpoint_api,
        api_key=_get_required_env("API_KEY_KINEXON"),
        username_basic=_get_required_env("USERNAME_KINEXON_SESSION"),
        password_basic=_get_required_env("PASSWORD_KINEXON_SESSION"),
        username_main=_get_required_env("USERNAME_KINEXON_MAIN"),
        password_main=_get_required_env("PASSWORD_KINEXON_MAIN"),
        endpoint_session=endpoint_session,
        endpoint_main=endpoint_main,
        timeout=10000,
    )
    return api


def get_competition_and_season(api):
    """Resolve competition and season ids used by the notebook.

    Returns (competition_id, season_id)
    """
    comp_id = api.get_competition_id_by_name(NAME_COMPETITION)
    if not comp_id:
        raise ValueError(f"Competition '{NAME_COMPETITION}' not found.")
    season_id = api.get_season_id_by_year(
        competition_id=comp_id, season_year=YEAR_SEASON
    )
    if not season_id:
        raise ValueError(
            f"Season '{NAME_SEASON}' not found in competition '{NAME_COMPETITION}'."
        )
    return comp_id, season_id


if __name__ == "__main__":
    # quick smoke test when run directly
    con = get_duckdb_connection()
    api = get_api_sportradar()
    print("Config ready", con, api)
