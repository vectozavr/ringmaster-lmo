# param-free-ringmaster

This branch contains the Nanochat Ringmaster/Muon experiment code.

## Files

- `ringmaster.py` runs the async Nanochat comparison and tuning experiments.
- `measure_nanochat_gradient_time.py` measures per-gradient GPU runtime for delay calibration.
- `plot_nanochat_traces.py` replots saved Nanochat traces.
- `nanochat_async.py` contains the Nanochat language-model objective.
- `asynchronous/` contains the async transport and optimizer implementations used by `ringmaster.py`.

## Setup

Install dependencies from `requirements.txt`.
