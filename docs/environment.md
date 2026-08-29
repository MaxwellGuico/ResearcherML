# Phase 0 — Reproducible Environment

Phase 0 was completed on 29 August 2026 (Asia/Singapore). This record establishes the environment and immutable benchmark reference before the autonomous research system is implemented.

## Environment

- Operating system: Windows 11 (`10.0.26200`), AMD64
- Processor: Intel64 Family 6 Model 189, 8 logical CPUs
- Python: `3.14.2`
- Virtual environment: repository-local `.venv`
- NumPy: `2.5.2`
- PyTorch: `2.13.0+cpu`
- PyTorch CUDA available: `False`
- Git reference at verification: `6566815d0f15ed98a34a82170fe113094b8bc3fd`

No NVIDIA tooling was detected, so the official CPU build of PyTorch was selected. The environment is intentionally local to the repository and ignored by Git.

## Recreate the environment

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `.\.venv\Scripts\python.exe` for project commands so the global Python installation is not used accidentally.

## Canonical data path

All future controllers, runners, PyTorch datasets, and models must consume data through `data.py`. Raw KuaiRand CSV access remains centralized there.

Verified split sizes:

- Train: 1,141,112 rows
- Validation: 124,909 rows
- Test: 170,588 rows

The `KuaiRand-Pure/` directory is ignored by Git, and no dataset files are tracked.

## Baseline verification

Command:

```powershell
.\.venv\Scripts\python.exe baseline.py --model fm --seed 0
```

The official NumPy FM stopped after epoch 11 and reproduced the recorded seed-0 reference:

- Validation: GAUC `0.6671`, nDCG@5 `0.5358`, primary `0.6015`
- Reference test: GAUC `0.6621`, nDCG@5 `0.5286`, primary `0.5953`

The test result was produced only because the immutable baseline command reports both splits. It is a reference reproduction, not an experiment-selection result. Future research iterations must select candidates using validation only.

## Source fingerprints

SHA-256 at verification:

```text
baseline.py  C8F7FC60178413E247E78BB231E7550EEEF52101B6493FCF1A4D2B0E5FE18F8A
data.py      1BF54F5F3A9F590EAB2F87F09A3C27422031867A20A5328D56CBD8C7DB36E541
evaluate.py  ECFDE28392EB14FEC4F488083251DF50624E1AF2B86278B962DAECFB42D195DE
submit.py    AB01BB2B970AE2A9F2EAD299F5240B71FF4126C2D9BB0E0C4DE6C7E245DC148C
```

## Dataset fingerprints

SHA-256 at verification:

```text
log_random_4_22_to_5_08_pure.csv          60B80994DA969CD53DA4D50C37BA3DAFD6FB185DF804C92C8410DF34845A9D2C
log_standard_4_08_to_4_21_pure.csv        5BB6EB0B3D9F47E5436CB5DC82EE1899B845EBF9750A5560B801E929E18BD41C
log_standard_4_22_to_5_08_pure.csv        429E3B948828942E572F2C3A5BE5A25799FFE75591D22D18CF417B9B534D31FD
user_features_pure.csv                    DC729A656301B4C6D07F713FE41D05EC9BFAAB670B90E531C70037CAF033C011
video_features_basic_pure.csv             A6F7EE02684C5777422306CDC416E170302288AA89ACA9DFEA995EDBD625BCC2
video_features_statistic_pure.csv         D5C9E237EF2C6C1FC0E7F27E952F215D6626ECD934B01A6C53ECFCC72540F6B6
```

## Git safety

The ignore rules protect:

- datasets and downloaded archives
- `.env` secrets and local virtual environments
- Python caches
- run, log, and checkpoint directories
- generated submissions
- `.ckpt`, `.npz`, `.pt`, and `.pth` model artifacts

No model architecture, training behavior, evaluator logic, or dataset preprocessing was changed during Phase 0.
