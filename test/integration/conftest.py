from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from hbl_etl_dagster.utils.api_helper import get_api_kinexon, get_api_sportradar


load_dotenv()


def _missing(required: list[str]) -> list[str]:
    return [k for k in required if not os.getenv(k)]


@pytest.fixture(scope="session")
def sportradar_api():
    required = [
        "BASE_URL",
        "AUTH_URL",
        "CLIENT_ID",
        "CLIENT_SECRET",
        "CLIENT_ORGANIZATION_ID",
    ]
    missing = _missing(required)
    if missing:
        pytest.skip(f"Missing Sportradar env vars: {missing}")
    return get_api_sportradar()


@pytest.fixture(scope="session")
def kinexon_api():
    required = [
        "ENDPOINT_KINEXON_SESSION",
        "ENDPOINT_KINEXON_MAIN",
        "API_KEY_KINEXON",
        "USERNAME_KINEXON_SESSION",
        "PASSWORD_KINEXON_SESSION",
        "USERNAME_KINEXON_MAIN",
        "PASSWORD_KINEXON_MAIN",
    ]
    missing = _missing(required)
    if missing:
        pytest.skip(f"Missing Kinexon env vars: {missing}")
    return get_api_kinexon()
