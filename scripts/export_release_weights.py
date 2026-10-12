#!/usr/bin/env python3
"""Convert training checkpoints into the released weight folders.

AE:    model.safetensors + config.yaml
Prior: model.safetensors (EMA weights) + config.yaml + latent_stats.json

Optimizer state, RNG state, data splits and file paths are not exported.
Usage (maintainers):
    python scripts/export_release_weights.py --kind ae    --id core-ae-liif-ct512 --checkpoint .../ae_checkpoint.pt --out weights/
    python scripts/export_release_weights.py --kind prior --id core-ldm-jit-ct512-liif --checkpoint .../best.pt --ae-id core-ae-liif-ct512 --out weights/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import yaml
from safetensors.torch import save_file

from c3d.models.ae.autoencoder_3d import canonical_head

PREPROCESSING_KEYS = ('ct_min_hu', 'ct_max_hu', 'ct_pad_hu', 'mri_percentile_low',
                      'mri_percentile_high', 'mri_foreground_threshold')
STATS_KEYS = ('latent_shape', 'normalization', 'scale', 'shift', 'formula',
              'per_channel_mean_raw', 'per_channel_variance_raw',
              'per_channel_mean_normalized', 'per_channel_variance_normalized')
SPACING = {'ct': [1.0, 1.0, 1.5], 'mri': None}   # MRI: spacing of the prepared IXI volumes was not recorded
# Axis order of the training arrays: CT D,H,W = z,y,x; the prepared IXI arrays are x,y,z.
ARRAY_AXES = {'ct': 'zyx', 'mri': 'xyz'}


def _tensors(state):
    return {k: v.detach().to('cpu').contiguous().clone() for k, v in state.items()}


def export_ae(checkpoint, out, model_id):
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = c['config']
    model = dict(cfg['model'])
    model['head'] = {**model['head'], 'name': canonical_head(model['head']['name'])}
    modality = cfg['experiment']['modality']
    release = {
        'id': model_id,
        'modality': modality,
        'native_shape': list(cfg['experiment']['native_shape']),
        'voxel_spacing_xyz_mm': SPACING[modality],
        'array_axes': ARRAY_AXES[modality],
        'preprocessing': {k: cfg['data'][k] for k in PREPROCESSING_KEYS if k in cfg['data']},
        'model': model,
    }
    state = {k[7:] if k.startswith('module.') else k: v for k, v in c['model'].items()}
    out.mkdir(parents=True, exist_ok=True)
    save_file(_tensors(state), str(out / 'model.safetensors'))
    (out / 'config.yaml').write_text(yaml.safe_dump(release, sort_keys=False))


def export_prior(checkpoint, out, model_id, ae_id):
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = c['config']
    state = dict(c['model'])
    state.update(c['ema']['shadow'])          # EMA weights, as used for sampling
    tr = cfg['training']
    release = {
        'id': model_id,
        'autoencoder': ae_id,
        'model': dict(cfg['model']),
        'objective': dict(cfg['objective']),
        'sampling': {'precision': tr.get('precision', 'bf16'),
                     'steps': int(tr.get('sample_steps', 50)),
                     'method': tr.get('sampling_method', 'heun')},
        'ema_decay': float(c['ema']['decay']),
        'training_step': int(c.get('step', 0)),
    }
    stats = {k: c['latent_stats'][k] for k in STATS_KEYS}
    out.mkdir(parents=True, exist_ok=True)
    save_file(_tensors(state), str(out / 'model.safetensors'))
    (out / 'config.yaml').write_text(yaml.safe_dump(release, sort_keys=False))
    (out / 'latent_stats.json').write_text(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['ae', 'prior'], required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ae-id')
    p.add_argument('--out', default='weights')
    a = p.parse_args()
    out = Path(a.out) / a.id
    if a.kind == 'ae':
        export_ae(a.checkpoint, out, a.id)
    else:
        if not a.ae_id:
            p.error('--ae-id is required for a prior')
        export_prior(a.checkpoint, out, a.id, a.ae_id)
    print('wrote', out)


if __name__ == '__main__':
    main()
