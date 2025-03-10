from typing import List, Dict, Any, Optional


class TimelineEvent:
    """Represents a single event in a timeline, capturing details such as type, time, and related players."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.id: int = data.get("id")
        self.type: str = data.get("type")
        self.time: str = data.get("time")
        self.match_time: Optional[int] = data.get("match_time")
        self.match_clock: Optional[str] = data.get("match_clock")
        self.competitor: Optional[str] = data.get("competitor")
        self.player: Optional[Dict[str, str]] = data.get("player")
        self.players: Optional[List[Dict[str, str]]] = data.get("players", [])
        self.home_score: Optional[int] = data.get("home_score")
        self.away_score: Optional[int] = data.get("away_score")
        self.scorer: Optional[Dict[str, str]] = data.get("scorer")
        self.assists: Optional[List[Dict[str, str]]] = data.get("assists", [])
        self.method: Optional[str] = data.get("method")
        self.zone: Optional[str] = data.get("zone")
        self.shot_type: Optional[str] = data.get("shot_type")
        self.outcome: Optional[str] = data.get("outcome")
        self.suspension_minutes: Optional[int] = data.get("suspension_minutes")

        # Extract players information
        self.name_player: Optional[str] = None
        self.id_str_player: Optional[str] = None
        self.id_player: Optional[int] = None
        self.name_blocker: Optional[str] = None
        self.id_str_blocker: Optional[str] = None
        self.id_blocker: Optional[int] = None
        self.name_goalkeeper: Optional[str] = None
        self.id_str_goalkeeper: Optional[str] = None
        self.id_goalkeeper: Optional[int] = None
        self.name_assist: Optional[str] = None
        self.id_str_assist: Optional[str] = None
        self.id_assist: Optional[int] = None

        self._extract_player_info()
        self._convert_id_str_to_int()

    def _extract_player_info(self) -> None:
        """Extracts player information from the event data."""
        if self.players:
            for player in self.players:
                if player["type"] == "shot":
                    self.name_player = player["name"]
                    self.id_str_player = player["id"]
                elif player["type"] == "blocked":
                    self.name_blocker = player["name"]
                    self.id_str_blocker = player["id"]
                elif player["type"] in {"saved", "goalkeeper"}:
                    self.name_goalkeeper = player["name"]
                    self.id_str_goalkeeper = player["id"]

        if self.scorer:
            self.name_player = self.scorer.get("name")
            self.id_str_player = self.scorer.get("id")

        if self.assists:
            self.id_str_assist = self.assists[0].get("id")
            self.name_assist = self.assists[0].get("name")

    def _convert_id_str_to_int(self) -> None:
        """Converts string player IDs to integers."""
        if self.id_str_player:
            self.id_player = int(self.id_str_player.split(":")[-1])
        if self.id_str_blocker:
            self.id_blocker = int(self.id_str_blocker.split(":")[-1])
        if self.id_str_goalkeeper:
            self.id_goalkeeper = int(self.id_str_goalkeeper.split(":")[-1])
        if self.id_str_assist:
            self.id_assist = int(self.id_str_assist.split(":")[-1])

    def __repr__(self) -> str:
        return (
            f"TimelineEvent(id={self.id}, type={self.type}, time={self.time}, match_time={self.match_time}, "
            f"competitor={self.competitor}, home_score={self.home_score}, "
            f"away_score={self.away_score}, name_player={self.name_player}, id_player={self.id_player}, "
            f"id_goalkeeper={self.id_goalkeeper}, name_goalkeeper={self.name_goalkeeper})"
        )
