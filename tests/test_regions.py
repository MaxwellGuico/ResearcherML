from __future__ import annotations

import unittest

from research_agent.planner import EvidencePlanner
from research_agent.regions import SearchRegionManager
from research_agent.review import ReviewDecision
from research_agent.state import ResearchState


class SearchRegionManagerTests(unittest.TestCase):
    def test_regions_preserve_independent_direction_history(self) -> None:
        manager = SearchRegionManager()
        history = [
            {"direction_id": "pointwise_fm_optimization", "search_region_id": "region_pointwise_fm_optimization", "metrics": {"primary": 0.61}, "decision": "accepted"},
            {"direction_id": "pairwise_fm_ranking", "search_region_id": "region_pairwise_fm_ranking", "metrics": {"primary": 0.59}, "decision": "rejected"},
        ]
        snapshots = manager.snapshots(history)
        self.assertEqual([snapshot.direction_id for snapshot in snapshots], ["pairwise_fm_ranking", "pointwise_fm_optimization"])
        self.assertEqual(snapshots[1].status, "PROMISING")

    def test_restart_creates_a_distinct_region(self) -> None:
        manager = SearchRegionManager()
        direction = EvidencePlanner().propose([], ResearchState())
        search_state = manager.choose_search_state(
            direction,
            [],
            ResearchState(),
            ReviewDecision("restart", "plateau"),
        )
        self.assertEqual(search_state.status, "RESTARTING")
        self.assertIn("restart", search_state.region_id)
        self.assertEqual(search_state.strategy, "diverse_restart")

