from typing import List, Dict, Any, Optional


class SportEventContext:
    """
    Holds contextual details about the sport event:
    sport, category, competition, season, stage, round, and groups.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        # Sport
        sport_data = data.get("sport", {})
        self.sport_id: str = sport_data.get("id", "")
        self.sport_name: str = sport_data.get("name", "")

        # Category
        category_data = data.get("category", {})
        self.category_id: str = category_data.get("id", "")
        self.category_name: str = category_data.get("name", "")
        self.country_code: str = category_data.get("country_code", "")

        # Competition
        competition_data = data.get("competition", {})
        self.competition_id: str = competition_data.get("id", "")
        self.competition_name: str = competition_data.get("name", "")
        self.competition_gender: str = competition_data.get("gender", "")

        # Season
        season_data = data.get("season", {})
        self.season_id: str = season_data.get("id", "")
        self.season_name: str = season_data.get("name", "")
        self.season_start_date: str = season_data.get("start_date", "")
        self.season_end_date: str = season_data.get("end_date", "")
        self.season_year: str = season_data.get("year", "")
        self.season_competition_id: str = season_data.get("competition_id", "")

        # Stage
        stage_data = data.get("stage", {})
        self.stage_order: Optional[int] = stage_data.get("order")
        self.stage_type: str = stage_data.get("type", "")
        self.stage_phase: str = stage_data.get("phase", "")
        self.stage_start_date: str = stage_data.get("start_date", "")
        self.stage_end_date: str = stage_data.get("end_date", "")
        self.stage_year: str = stage_data.get("year", "")

        # Round
        round_data = data.get("round", {})
        self.round_number: Optional[int] = round_data.get("number")

        # Groups
        self.groups: List[Dict[str, str]] = data.get("groups", [])

    def __repr__(self) -> str:
        return f"SportEventContext(sport={self.sport_name}, category={self.category_name}, competition={self.competition_name}, season={self.season_name}, stage={self.stage_type}, round={self.round_number})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sport_id": self.sport_id,
            "sport_name": self.sport_name,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "country_code": self.country_code,
            "competition_id": self.competition_id,
            "competition_name": self.competition_name,
            "competition_gender": self.competition_gender,
            "season_id": self.season_id,
            "season_name": self.season_name,
            "season_start_date": self.season_start_date,
            "season_end_date": self.season_end_date,
            "season_year": self.season_year,
            "season_competition_id": self.season_competition_id,
            "stage_order": self.stage_order,
            "stage_type": self.stage_type,
            "stage_phase": self.stage_phase,
            "stage_start_date": self.stage_start_date,
            "stage_end_date": self.stage_end_date,
            "stage_year": self.stage_year,
            "round_number": self.round_number,
            "groups": ", ".join([str(group) for group in self.groups]),
        }
