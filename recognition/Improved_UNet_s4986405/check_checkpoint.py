"""Inspect one saved run and check a strict model reload on CPU."""
import argparse
import csv
import json
from pathlib import Path

import torch

from modules import UNet2D
from training_io import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.run_dir.expanduser().resolve()
    last = load_checkpoint(directory / 'last.pt')
    best = load_checkpoint(directory / 'best.pt')
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    with (directory / 'history.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(history) != last['epoch'] or len(rows) != len(history):
        raise RuntimeError('Epoch/history row count mismatch')
    if history != last['history']:
        raise RuntimeError('Checkpoint and JSON histories differ')
    best_entry = min(history, key=lambda entry: entry['validation']['loss'])
    if best['epoch'] != best_entry['epoch'] or last['best_epoch'] != best_entry['epoch']:
        raise RuntimeError('Best checkpoint does not match minimum validation loss')
    if best['architecture'] != 'UNet2D':
        raise ValueError('This inspection script currently supports baseline UNet2D')
    model = UNet2D(**best['model_config']).cpu()
    model.load_state_dict(best['model_state'], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 32, 16))
    if tuple(logits.shape) != (1, 6, 32, 16) or not torch.isfinite(logits).all():
        raise RuntimeError('Invalid reloaded model output')
    print('Run directory:', directory)
    print('History epochs:', len(history))
    print('Latest epoch:', last['epoch'])
    print('Best epoch:', best['epoch'])
    print('Best validation loss:', last['best_val_loss'])
    print('Partial run:', last['config']['partial_run'])
    print('History consistency: PASS')
    print('Checkpoint reload: PASS')


if __name__ == '__main__':
    main()
