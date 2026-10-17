# GIFT-Eval feasibility probe

This is a partial integration assessment, not a 97-task leaderboard submission.
See [the detailed assessment](../../docs/gift-eval.zh-CN.md).

- Four complete tasks reproduce the published Timestral-5M metrics on CPU: hospital monthly short,
  ETT1 hourly short/long, and CloudOps l2c hourly short with past covariates.
- All eleven metrics match within rtol=1e-3 / atol=1e-5; maximum observed relative error is about 1.02e-6.
- Hospital short and ETT1 long also match on MPS within the same tolerances.
- Four LoRA steps on 32 hospital training series connect the training and official evaluation paths.
  This is a correctness smoke check, not evidence of adaptation gains.
- Short-history eligibility, the full 97-task horizon inventory, source revisions and dependency versions are retained.

`probe.py` and `probe_mps.py` are source snapshots of the local probe. They execute cells from the pinned
official Timestral notebook and use the cache paths recorded in the assessment; they are not production
integration commands and require fresh training output paths for an independent rerun.
Raw Arrow datasets, downloaded source checkouts, the isolated environment and generated weights stay under ignored `.cache`.
