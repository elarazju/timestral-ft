"""Real 5M acceptance includes missingness, dynamic roles and both category types."""
import os

import pytest
import torch
from test_dhf_acceptance import require_backend

from timestral_ft import TrainConfig
from timestral_ft.augmentation import augment_sequence
from timestral_ft.compat import load_model
from timestral_ft.dhf import dhf_pinball_loss, differentiable_dhf_forecast
from timestral_ft.lora import apply_lora


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('TIMESTRAL_INTEGRATION') != '1', reason='Set TIMESTRAL_INTEGRATION=1')
@pytest.mark.parametrize('device', ['cpu', 'mps', 'cuda'])
@pytest.mark.parametrize('mode', ['full', 'lora'])
@pytest.mark.parametrize('categories', [False, True])
def test_real_5m_augmented_categorical_forward_backward(device, mode, categories):
    require_backend(device)
    model = load_model('mistralai/Timestral-5M', revision='bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f', device=device)
    if mode == 'lora':
        apply_lora(model, rank=2, alpha=4)
    x = torch.randn(1, 4, 160, device=device)
    cfg = TrainConfig(missing_probability=1, missing_max_span=16, dynamic_role_sampling=True)
    aug = augment_sequence(x, torch.zeros(1, 4, dtype=torch.long), cfg,
                           torch.Generator().manual_seed(42), torch.Generator().manual_seed(43))
    cats = (torch.arange(160, dtype=torch.float64) % 4 + 2 ** 35)[None, None]
    layout = torch.tensor([[-1, 1, 1, 3, 3]])
    category_args = dict(categorical_past_covariates=cats, categorical_known_covariates=cats) if categories else {}
    predicted = differentiable_dhf_forecast(model, aug.inputs, layout, labels=aug.labels,
        numeric_role_ids=aug.role_ids, **category_args)
    loss = dhf_pinball_loss(predicted, aug.labels, model.config.quantile_knots)
    loss.backward()
    assert torch.isfinite(loss)
    assert (predicted.supervision_mask & aug.hidden_mask).any()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    previous = {name: value.detach().clone() for name, value in model.named_parameters()}
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4).step()
    assert any(not torch.equal(value, previous[name]) for name, value in model.named_parameters())
