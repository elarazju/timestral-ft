# Contributing

This independent community project welcomes bug reports, tests, and focused contributions.
Please include the base model revision, timestral-ft version, device, minimal configuration,
and a small synthetic reproduction. Do not include private time series or access tokens.

```sh
uv sync --locked --extra dev --python 3.12
uv run pytest -q
uv run ruff check src tests scripts
uv run python -m build
```

The default suite must stay offline with respect to model weights. Run the pinned real-weight test explicitly:

```sh
TIMESTRAL_INTEGRATION=1 uv run pytest -q -m integration
```

Changes to the upstream compatibility layer must preserve official inference parity and finite gradients.
LoRA changes must preserve zero initial delta, frozen base weights, and merge equivalence.
Data changes must include tests for time boundaries and future-target isolation.

CPU and MPS are locally validated; CUDA and other model sizes must not be described as validated without
corresponding hardware results. Accelerator tests run automatically when hardware is available.
Require a backend explicitly with `TIMESTRAL_TEST_DEVICE=cuda pytest -q tests/test_devices.py`
(or `mps`); unavailable required hardware fails the test job instead of giving a green skip.
Do not tune example hyperparameters using the held-out test set. Publish regressions as well as improvements.

Contributions are licensed under the repository's Apache-2.0 license. Model weights retain their own licenses.
