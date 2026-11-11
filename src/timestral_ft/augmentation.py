"""Explicit input corruption and synthetic information-permission tasks.

Only numeric variables are reassigned. Sampling is per example, not per DHF
block, so every block has a consistent role firewall. These distributions are
project choices; the report does not disclose its exact sampling recipe.
"""
from dataclasses import dataclass

import torch
from timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET


@dataclass(frozen=True)
class AugmentedSequence:
    inputs: torch.Tensor
    labels: torch.Tensor
    input_mask: torch.Tensor
    original_mask: torch.Tensor
    role_ids: torch.Tensor
    hidden_mask: torch.Tensor

    def statistics(self):
        return {"original_observed_points": int(self.original_mask.sum()),
                "input_observed_points": int(self.input_mask.sum()),
                "artificially_hidden_points": int(self.hidden_mask.sum()),
                "sampled_target_rows": int((self.role_ids == ROLE_TARGET).sum()),
                "sampled_past_rows": int((self.role_ids == ROLE_PAST_COV).sum()),
                "sampled_known_rows": int((self.role_ids == ROLE_KNOWN_COV).sum())}


def sample_roles(batch, variables, probabilities, generator):
    """Draw iid roles then uniformly repair examples that have no target."""
    choices = torch.multinomial(torch.tensor(probabilities, dtype=torch.float64),
                                batch * variables, replacement=True, generator=generator).reshape(batch, variables)
    for row in choices:
        if not (row == 0).any():
            row[int(torch.randint(variables, (), generator=generator))] = 0
    return torch.tensor([ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV])[choices]


def augment_sequence(sequence, role_ids, cfg, missing_generator, role_generator, *, candidate_count=None):
    """Hide at most one contiguous interval per numeric row, without editing labels.

    Natural NaNs and artificially hidden finite values remain distinguishable.
    An entirely hidden prefix can remove that block's supervision eligibility;
    no observed point is fabricated to keep a training example alive.
    """
    original = torch.isfinite(sequence)
    hidden = torch.zeros(sequence.shape, dtype=torch.bool)
    if cfg.missing_probability:
        b, v, length = sequence.shape
        for i in range(b):
            for j in range(v):
                if float(torch.rand((), generator=missing_generator)) < cfg.missing_probability:
                    size = int(torch.randint(1, min(cfg.missing_max_span, length) + 1, (), generator=missing_generator))
                    start = int(torch.randint(length - size + 1, (), generator=missing_generator))
                    hidden[i, j, start:start + size] = True
    hidden = hidden.to(sequence.device) & original
    roles = role_ids.to(sequence.device).clone()
    if cfg.dynamic_role_sampling:
        candidates = sequence.shape[1] if candidate_count is None else candidate_count
        roles[:, :candidates] = sample_roles(sequence.shape[0], candidates, cfg.role_probabilities,
                                            role_generator).to(sequence.device)
    mask = original & ~hidden
    return AugmentedSequence(sequence.masked_fill(~mask, float('nan')), sequence,
                             mask, original, roles, hidden)


def prepare_augmented_batch(batch, cfg, missing_generator, role_generator):
    """Pool numeric inputs; only originally declared targets can change roles.

    Fixed covariates retain their declarations. Window past covariates have no
    future observations, and that suffix stays NaN.
    """
    is_dhf = 'sequence' in batch
    sequence = batch['sequence'] if is_dhf else torch.cat([batch['context'], batch['labels']], -1)
    arrays, roles = [sequence], [ROLE_TARGET] * sequence.shape[1]
    for key, role in [('past_covariates', ROLE_PAST_COV), ('known_covariates', ROLE_KNOWN_COV)]:
        if key in batch:
            values = batch[key]
            values = torch.nn.functional.pad(values, (0, sequence.shape[-1] - values.shape[-1]), value=float('nan'))
            arrays.append(values)
            roles.extend([role] * values.shape[1])
    numeric = torch.cat(arrays, 1)
    role_ids = torch.tensor(roles).expand(numeric.shape[0], -1)
    return augment_sequence(numeric, role_ids, cfg, missing_generator, role_generator,
                            candidate_count=sequence.shape[1])
