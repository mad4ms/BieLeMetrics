from typing import List, Dict, Any, Optional

from src.models.helper_sportradar.TimelineEvent import (
    TimelineEvent,
)


class EventTimeline:
    """Organizes the timeline of a sporting event, capturing all events in sequence."""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.events: List[TimelineEvent] = [
            TimelineEvent(event) for event in data
        ]
        # Due to the nature of the data, we need to add scores to score_change events
        self._add_goalkeepers_to_score_changes()
        # Add scores to all events
        self._add_scores_to_events()

    def __repr__(self) -> str:
        return f"EventTimeline(events={len(self.events)} events)"

    def get_events_by_type(self, event_type: str) -> List[TimelineEvent]:
        """Filters and returns events of a specific type."""
        return [event for event in self.events if event.type == event_type]

    def get_score_changes(self) -> List[TimelineEvent]:
        """Returns all score change events."""
        return self.get_events_by_type("score_change")

    def get_events_by_player(self, player_id: str) -> List[TimelineEvent]:
        """Returns all events involving a specific player."""
        return [
            event
            for event in self.events
            if event.player
            and event.player.get("id") == player_id
            or any(p.get("id") == player_id for p in event.players)
        ]

    def _add_goalkeepers_to_score_changes(self) -> None:
        """Adds goalkeeper information to score_change events by analyzing preceding events."""
        for i, event in enumerate(self.events):
            if (event.type == "score_change" and not event.id_goalkeeper) or (
                event.type == "shot_blocked" and not event.id_goalkeeper
            ):
                # Search for preceding shot-related events
                for j in range(i - 1, -1, -1):
                    preceding_event = self.events[j]
                    if preceding_event.type in {
                        "shot_saved",
                        "shot_off_target",
                        "shot_blocked",
                    }:
                        # Ensure the competitor is the opposing team
                        if preceding_event.competitor != event.competitor:
                            event.name_goalkeeper = (
                                preceding_event.name_goalkeeper
                            )
                            event.id_goalkeeper = preceding_event.id_goalkeeper
                            break

    def _add_scores_to_events(self) -> None:
        """Adds score information to score_change events by analyzing preceding events."""
        home_score = 0
        away_score = 0
        for event in self.events:
            if event.type == "score_change":
                if event.competitor == "home":
                    home_score += 1
                else:
                    away_score += 1
            event.home_score = home_score
            event.away_score = away_score
