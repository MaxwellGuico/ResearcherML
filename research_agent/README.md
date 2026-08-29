# Research Agent Package

This package implements the autonomous research system in small phases. It is
separate from the immutable official baseline and evaluator.

## Fixed boundaries

- Load canonical prepared benchmark data only through `data.py`.
- Evaluate ranking quality only through the unchanged `evaluate.py`.
- Keep `baseline.py` as the official NumPy reference implementation.
- Implement candidate models and training loops with PyTorch.
- Select candidates using validation only; test is final confirmation.

## Component map

| Component | Responsibility | Implementation phase |
| --- | --- | --- |
| `contracts.py` | Fixed benchmark facts and safety boundaries | 1 |
| `logger.py` / `store.py` | Append-only events, experiment records, and artifacts | 2 |
| `metrics.py` | Prediction validation and official evaluator integration | 3 |
| `runner.py` | Isolated PyTorch candidate execution | 4 |
| `safety.py` | Deterministic contract, leakage, and resource checks | 5 |
| `controller.py` | Experiment lifecycle and accept/reject decisions | 6 |
| `models/` | Approved PyTorch candidate architectures | Later |
| `templates/` | Separately approved, versioned experiment configurations | Later |

Concrete experiment templates are deliberately not defined in this package
foundation. They are implementation configuration rather than architecture.
