"""Deterministic diffusion holdout and training-only latent statistics.

The AE export is already an AE training split. We reserve a second, fixed
holdout for diffusion and merge only the remaining raw-volume moments. Cache
arrays, cache stats and the AE validation/test split are never changed.
"""
from __future__ import annotations

import copy
import hashlib
import math

from torch.utils.data import Subset

from c3d.data.identity import digest
from c3d.data.latent_statistics import combine_moments


def prepare_data(dataset, config):
    validation = config.get('validation', {})
    fraction = float(validation.get('fraction', 0.0))
    if not 0 <= fraction < 1:
        raise ValueError('validation.fraction must be in [0,1)')
    seed = int(validation.get('seed', 91415))
    count = len(dataset)
    if fraction and count < 2:
        raise ValueError('A diffusion holdout needs at least two cached volumes')
    # Source paths align head comparisons when the caches export the same data.
    sources = dataset.manifest.get('sources', [])
    keys = [sources[i]['path'] if i < len(sources) else str(i) for i in range(count)]
    order = sorted(range(count), key=lambda i: hashlib.sha256(f'{seed}:{keys[i]}'.encode()).hexdigest())
    nval = min(count - 1, max(1, round(count * fraction))) if fraction else 0
    val_ids = order[:nval]
    train_ids = sorted(order[nval:])
    split = {'schema': 1, 'cache_identity': dataset.identity, 'seed': seed,
             'fraction': fraction, 'train_indices': train_ids, 'validation_indices': val_ids,
             'selection': 'sha256(seed:source_path)',
             'scope': 'diffusion holdout within the exported AE training split'}
    stats = copy.deepcopy(dataset.stats)
    if nval:
        if not dataset.on_the_fly:
            raise ValueError('Diffusion holdout requires a raw cache (schema 3); regenerate raw latents/stats')
        stats = combine_moments(dataset.records[i]['moments'] for i in train_ids)
        mode = dataset.manifest['normalization']
        if mode not in ('global_sigma', 'zscore'):
            raise ValueError('Unsupported normalization: ' + mode)
        scale = max(stats['global_std'], 1e-6)
        shift = stats['global_mean'] if mode == 'zscore' else 0.0
        stats.update(normalization=mode, scale=scale, shift=shift,
                     formula='z_norm = (z_raw - shift) / scale', storage='raw',
                     normalization_at='dataset_read', manifest_sha256=digest(dataset.manifest),
                     statistics_scope='diffusion_training_only', split_sha256=digest(split))
        stats['per_channel_mean_normalized'] = [(m-shift)/scale for m in stats['per_channel_mean_raw']]
        stats['per_channel_variance_normalized'] = [v/scale**2 for v in stats['per_channel_variance_raw']]
    if not math.isfinite(stats['scale']) or stats['scale'] <= 0:
        raise ValueError('Invalid training normalization scale')
    # One immutable transformation shared by training, validation and decoding.
    dataset.stats = stats
    dataset.scale = float(stats['scale'])
    dataset.shift = float(stats.get('shift', 0.))
    dataset.identity = digest({'manifest': dataset.manifest, 'stats': stats})
    return Subset(dataset, train_ids), Subset(dataset, val_ids), split
