"""Run: python examples/report_round6/encode.py --device cpu (or mps)."""

import argparse

import torch

from timestral_ft import encode
from timestral_ft.compat import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    model = load_model("mistralai/Timestral-5M", revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f", device=args.device)
    # Two examples, two numeric target variables, right-padded storage.
    x = torch.sin(torch.arange(150) / 8).repeat(2, 2, 1)
    x[0, 1] = float("nan")
    result = encode(model, x, lengths=[130, 150], pooling="mean")
    print("patches:", tuple(result.patches.shape), "pooled:", tuple(result.pooled.shape))
    print("valid variables:", result.valid.cpu().tolist())
    print("first-example patch starts:", result.patch_start[0].cpu().tolist())
    print("first-example patch ends:", result.patch_end[0].cpu().tolist())
    assert torch.isfinite(result.patches).all()
    assert not result.valid[0, 1]


if __name__ == "__main__":
    main()
