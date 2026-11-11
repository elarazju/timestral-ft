# Round three: missing observations and variable roles

Run from the repository root after installing the project. Published evidence is
immutable; new experiments need fresh output/results directories:

```sh
python scripts/run_report_round3.py --phase preflight --output runs/my-augmentation --results results/my-augmentation
python scripts/run_report_round3.py --phase run --output runs/my-augmentation --results results/my-augmentation
python scripts/audit_report_round3.py --output runs/my-augmentation --results results/my-augmentation
```

Preflight measures ten steps for each arm without test scoring. It chooses the
largest common 200/100/50-step schedule fitting the eight-hour planning estimate
(including 40% allowance); this is not a hard runtime timeout. The complete matrix
keeps ETTh1/ETTh2 and seeds 42/43/44 paired. All four arms use normalized DHF loss,
448-point sequences, initial 32-point prefix, maximum six-patch forecast blocks,
LoRA rank 8/alpha 16, learning rate 1e-4 and batch 4 × accumulation 4.

The arms are `baseline`, `missing`, `roles`, and `both`. Missingness independently
selects each numeric row with probability .5, then hides one interval with uniform
length 1–64 and uniform valid start. Original finite labels remain available for
supervision; an entirely unobserved target prefix makes that target/block ineligible.
Only originally declared target columns are role candidates. Probabilities are
[.5 target, .25 past-only, .25 known-future], with a uniformly chosen target restored
if an example has none. Declared covariates never change roles. These distributions
are project choices, not a disclosed official training recipe.

Ordinary uncorrupted validation at H192/stride192 selects every model. Only after
all 24 selected weight checksums have been frozen do we score validation and test
at H96/H192, with either complete 256-point history or a fixed gap at indices
[128,192) in every target. Targets outside the history are never removed from scoring.
The forecast arrays include the actual input contexts for auditing. Seven metrics,
separate RNG states, supervision/missingness/role counts, timing, all model hashes
and sampled memory counters are retained. Equal steps do not equalize scored labels.

For application data, enable the mechanisms with:

```python
from timestral_ft import TrainConfig

training = TrainConfig(
    mode="lora", training_strategy="dhf", loss_space="normalized",
    validation_loss_space="history_scaled", context_length=256, prediction_length=192,
    dhf_max_block_patches=6, missing_probability=.5, missing_max_span=64,
    dynamic_role_sampling=True, role_probabilities=[.5, .25, .25],
)
```

Both mechanisms default off. Full-parameter training, LoRA, window training,
exact checkpoint resume and official-compatible merged exports remain supported.
Window mode only hides values in its available inputs; hiding future label slots
has no forward effect. DHF can also hide later context tokens while retaining
those positions as labels for earlier forecast blocks.

DHF categories use a **per-example, per-origin reference path**: target encoding
reads only that origin's target history, while known covariates are cropped to the
current block's endpoint. It makes multiple Transformer calls and is not a shared
encoding speedup. Numeric-only DHF keeps the shared path and its whole-window
known-covariate normalization. Categorical preprocessing is discrete; gradients
through category codes or their target-encoding inputs are not supported, while
model/adapter parameter gradients are supported and tested.

Weights, inputs, predictions, checkpoints and source snapshots stay in ignored
`runs/`. The lightweight evidence lives in `results/report_round3/`. To reproduce
older rounds after later source changes, use their recorded source snapshots or
commits, not a modified current checkout.

The pinned upstream encoder uses target encoding only when an example has one
current target. With multiple targets it passes numeric category codes instead;
dynamic roles may change which branch applies. The table processor uses a
train-only compact vocabulary. CPU float64 preserves raw category identities
until encoding; the upstream multi-target fallback casts numeric codes to FP32,
so arbitrary huge external codes are not guaranteed distinct in that branch.
