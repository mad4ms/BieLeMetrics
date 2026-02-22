"""Legacy sensor module.

Sensors are currently not part of the active Dagster Definitions setup.
This module keeps a compatibility symbol (`fixture_sensor`) so imports from
older tooling do not crash.
"""

from dagster import SkipReason, sensor


@sensor(name="fixture_sensor", minimum_interval_seconds=60)
def fixture_sensor(_context):
    """No-op compatibility sensor (currently unused)."""
    yield SkipReason("fixture_sensor is disabled in the current pipeline setup")
