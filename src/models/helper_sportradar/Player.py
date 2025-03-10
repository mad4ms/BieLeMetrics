from typing import Dict, Any


class PlayerStats:
    """
    Stores individual player statistics.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.player_id: str = data.get("id", "")
        self.name: str = data.get("name", "")
        self.statistics: Dict[str, Any] = data.get("statistics", {})

    def __repr__(self) -> str:
        return f"PlayerStats(name={self.name}, player_id={self.player_id})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "statistics": self.statistics,
        }
