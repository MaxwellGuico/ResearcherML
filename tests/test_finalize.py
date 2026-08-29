from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.finalize import _selected_iteration
from research_agent.store import ArtifactStore


class FinalizationTests(unittest.TestCase):
    def test_baseline_is_selected_when_no_agent_candidate_was_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            self.assertIsNone(_selected_iteration(store, "baseline"))

    def test_missing_selected_experiment_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                _selected_iteration(store, "exp_999")

