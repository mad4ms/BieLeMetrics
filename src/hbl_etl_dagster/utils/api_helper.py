"""Configuration and shared helpers for HBL ETL scripts.

This module extracts the top-level constants and creates helpers to
get a DuckDB connection and initialized API client. It's a direct
translation of the notebook's configuration cells.
"""

import datetime
import os

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


def get_duckdb_connection(db_path: str = None):
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
        base_url=os.getenv("BASE_URL", ""),
        auth_url=os.getenv("AUTH_URL", ""),
        client_id=os.getenv("CLIENT_ID", ""),
        client_secret=os.getenv("CLIENT_SECRET", ""),
        org_id=os.getenv("CLIENT_ORGANIZATION_ID"),
        scopes=["read:organization"],
        sport="handball",
    )
    return api


def get_api_kinexon():
    """Initialize and return the Kinexon API client using environment variables.

    Matches the notebook's KinexonAPI initialisation.
    """
    from kinexon_handball_api.handball import HandballAPI

    api = HandballAPI(
        base_url=os.getenv(
            "ENDPOINT_KINEXON_SESSION", "https://hbl-cloud.kinexon.com/api"
        ),
        api_key=os.getenv("API_KEY_KINEXON", "your_api_key_here"),
        username_basic=os.getenv("USERNAME_KINEXON_SESSION", "your_username_here"),
        password_basic=os.getenv("PASSWORD_KINEXON_SESSION", "your_password_here"),
        username_main=os.getenv("USERNAME_KINEXON_MAIN", "your_username_here"),
        password_main=os.getenv("PASSWORD_KINEXON_MAIN", "your_password_here"),
        endpoint_session=os.getenv(
            "ENDPOINT_KINEXON_SESSION",
            "https://hbl-cloud.kinexon.com/api/session",
        ),
        endpoint_main=os.getenv(
            "ENDPOINT_KINEXON_MAIN",
            "https://hbl-cloud.kinexon.com/api",
        ),
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
