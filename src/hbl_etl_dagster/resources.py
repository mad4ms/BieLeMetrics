# src/hbl_etl_dagster/resources.py

from dagster import resource
from src.hbl_etl_sportradar_kinexon.config import get_api_sportradar
from src.hbl_etl_sportradar_kinexon.config import get_api_kinexon


@resource
def sportradar_api(_):
    """
    Initializes the Sportradar API client once per run/step, using env vars etc.
    """
    return get_api_sportradar()


@resource
def kinexon_api(_):
    return get_api_kinexon()
