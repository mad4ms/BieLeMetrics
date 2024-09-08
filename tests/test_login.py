from bielemetrics_kinexon_api_wrapper import *

from sportradar import Handball

import pytest


def test_imports():
    assert load_credentials
    assert login
    assert authenticate
    assert make_api_request
    assert fetch_team_ids
    assert fetch_event_ids
    assert fetch_game_csv_data
    assert get_available_metrics_and_events
    assert Handball


# Run the test with the following command:
# pytest tests/test_imports.py
