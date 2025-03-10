from typing import Any, Dict, List


class EventStatus:
    """
    Stores event status information.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.status: str = data.get("status", "")
        self.match_status: str = data.get("match_status", "")
        self.home_score: int = data.get("home_score", 0)
        self.away_score: int = data.get("away_score", 0)
        self.winner_id: str = data.get("winner_id", "")
        self.period_scores: List[Dict[str, Any]] = data.get(
            "period_scores", []
        )

    def __repr__(self) -> str:
        return f"Status: {self.status}, Match Status: {self.match_status}, Home Score: {self.home_score}, Away Score: {self.away_score}, Winner ID: {self.winner_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "match_status": self.match_status,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "winner_id": self.winner_id,
            "period_scores": ",".join(
                [
                    f"{period['home_score']}:{period['away_score']}"
                    for period in self.period_scores
                ]
            ),
        }
