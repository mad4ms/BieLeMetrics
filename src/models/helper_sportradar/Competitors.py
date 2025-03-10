from typing import List, Dict, Any
import sys
import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.models.helper_sportradar.Player import PlayerStats


class CompetitorStats:
    """
    Collects team-level statistics and detailed player stats.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.abbreviation: str = data.get("abbreviation", "")
        self.qualifier: str = data.get("qualifier", "")
        self.statistics: Dict[str, Any] = data.get("statistics", {})
        self.players: List[PlayerStats] = [
            PlayerStats(player) for player in data.get("players", [])
        ]

    def __repr__(self) -> str:
        return f"CompetitorStats(id={self.id}, name={self.name}, abbreviation={self.abbreviation}, qualifier={self.qualifier}, statistics={self.statistics}, players={self.players})"
