# src/hbl_etl_dagster/resources.py

from dagster import resource

from src.hbl_etl_dagster.utils.api_helper import get_api_kinexon, get_api_sportradar


@resource
def sportradar_api(_):
    """
    Initializes the Sportradar API client once per run/step, using env vars etc.
    """
    return get_api_sportradar()


@resource
def kinexon_api(_):
    """
    Initializes the Kinexon API client once per run/step, using env vars etc.
    """
    return get_api_kinexon()
