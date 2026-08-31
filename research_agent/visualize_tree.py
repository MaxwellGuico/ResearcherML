"""Regenerate the visual research tree for an existing artifact directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from .logger import ResearchLogger
from .research_tree import ResearchTree
from .state import ResearchState
from .store import ArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Existing research-run artifact directory")
    args = parser.parse_args()
    store = ArtifactStore(args.artifact_dir)
    persisted = store.read_root_json("state.json") or {}
    state = ResearchState.from_dict(persisted) if persisted else ResearchState()
    tree = ResearchTree(ResearchLogger(store)).refresh(state)
    destination = Path(args.artifact_dir) / "research_tree.md"
    print(
        f"Wrote {destination} with {len(tree['hypotheses'])} hypotheses "
        f"and {len(tree['experiments'])} experiment branches."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
