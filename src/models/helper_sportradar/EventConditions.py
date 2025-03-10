from typing import Optional, Any, Dict, List

from src.models.helper_sportradar.Referee import Referee


class SportEventConditions:
    """
    Holds conditions around a sporting event such as referees, attendance, and ground.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        referees_data = data.get("referees", [])
        self.referees: List[Referee] = [Referee(r) for r in referees_data]

        attendance_data = data.get("attendance", {})
        self.attendance_count: Optional[int] = attendance_data.get("count")

        ground_data = data.get("ground", {})
        self.ground_neutral: bool = ground_data.get("neutral", False)

    def __repr__(self) -> str:
        return f"SportEventConditions(referees={self.referees}, attendance={self.attendance_count}, ground_neutral={self.ground_neutral})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "referees": [r.to_dict() for r in self.referees],
            "attendance_count": self.attendance_count,
            "ground_neutral": self.ground_neutral,
        }
