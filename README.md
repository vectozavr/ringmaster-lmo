# param-free-ringmaster

This branch contains the Nanochat Ringmaster/Muon experiment code.

## Files

- `code/experiments/ringmaster.py` runs the async Nanochat comparison and tuning experiments.
- `code/experiments/measure_nanochat_gradient_time.py` measures per-gradient GPU runtime for delay calibration.
- `code/experiments/plot_nanochat_traces.py` replots saved Nanochat traces.
- `code/experiments/nanochat_async.py` contains the Nanochat language-model objective.
- `code/experiments/asynchronous/` contains the async transport and optimizer implementations used by `ringmaster.py`.

## Setup

Install dependencies from `code/requirements.txt`.
