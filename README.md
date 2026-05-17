# param-free-ringmaster

This branch contains the Nanochat Ringmaster/Muon experiment code.

## Files

- `ringmaster.py` runs the async Nanochat comparison and tuning experiments.
- `measure_nanochat_gradient_time.py` measures per-gradient GPU runtime for delay calibration.
- `plot_nanochat_traces.py` replots saved Nanochat traces.
- `src/nanochat_async.py` contains the Nanochat language-model objective.
- `src/asynchronous/` contains the async transport and optimizer implementations used by `ringmaster.py`.
- `src/factory.py` and `src/signature.py` contain small helper utilities used by the experiment code.

## Setup

Install dependencies from `requirements.txt`.
