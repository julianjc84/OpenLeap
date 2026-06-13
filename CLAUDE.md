# OpenLeap

Open driver + hand tracking for the original Leap Motion Controller. Read
`README.md` first — it documents the device protocol and the pipeline stages.

## Build & test

```bash
( cd rust-driver && cargo build && cargo clippy --all-targets )  # driver
.venv/bin/python -m py_compile tracking/*.py                   # pipeline
./run_live.sh                # live viewer (polkit prompt; device must be plugged in)
```

No `#[allow(...)]`/lint suppressions — fix lints honestly.

## Live-test cycle

- `run_live.sh` stops the closed `ultraleap-hand-tracking-service` and runs
  `openleap stream` as root via pkexec, piping frames to the GTK viewer.
  Do NOT restart the closed service after tests — the user keeps it off.
- The device draws real power: the user keeps it unplugged when not testing.
  Ask before assuming it's connected.
- Offline analysis without the device: `tracking/recordings/<ts>/` (PGM
  frames + results.jsonl) can be replayed through `skeleton.run()` — this is
  how fixes are validated (see git history for the replay pattern).

## Conventions

- `skeleton.py` is the algorithm spec (future Rust port target). Mercury
  conventions (normalization, lastKeypoints, handedness mirror) must match
  Monado's `hg_model.cpp` — keep a local Monado checkout to cross-check.
- `hand-tracking-models/` is a separate git clone (gitignored); never vendor
  it into this repo.
- Reference clones (monado, mercury_train, leapuvc, librealuvc, the 2013
  OpenLeap) are kept as separate local checkouts — never vendored here.
