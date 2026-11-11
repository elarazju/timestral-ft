"""Small runnable API example; synthetic demonstration, not benchmark evidence."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from timestral_ft import DataConfig, TrainConfig, finetune


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='runs/augmentation-smoke')
    args = parser.parse_args()
    t = np.arange(1024)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=len(t), freq='h'),
                          'load': np.sin(t / 24) + rng.normal(0, .1, len(t)),
                          'temperature': np.cos(t / 48) + rng.normal(0, .1, len(t)),
                          'auxiliary': np.sin(t / 24 + .3) + rng.normal(0, .1, len(t))})
    artifact = finetune(frame.iloc[:768], frame.iloc[768:],
        revision='bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f',
        data_config=DataConfig(target_columns=['load', 'temperature', 'auxiliary'], frequency='h'),
        train_config=TrainConfig(mode='lora', training_strategy='dhf', loss_space='normalized',
            validation_loss_space='history_scaled', context_length=128, prediction_length=96,
            missing_probability=.5, missing_max_span=32, dynamic_role_sampling=True,
            max_steps=4, eval_every=2, warmup_steps=1, batch_size=2, gradient_accumulation=1,
            device=args.device), output_dir=args.output)
    prediction = artifact.predict(frame.iloc[-128:], prediction_length=96)
    prediction.to_csv(Path(args.output) / 'forecast.csv', index=False)
    print(Path(args.output) / 'forecast.csv')


if __name__ == '__main__':
    main()
