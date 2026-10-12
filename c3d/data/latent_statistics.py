"""Raw latent IO and CPU-only moment aggregation. No torch/model imports."""
from __future__ import annotations
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np

from c3d.data.identity import atomic_json, digest, sha256, verify_ae_manifest
from c3d.data.latent_progress import log, stage


@contextlib.contextmanager
def file_lock(path, *, shared=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NFS record-lock emulation requires read access for shared locks.
    with path.open('a+') as handle:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(handle, operation | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def moments(array):
    """Population moments of one C,D,H,W raw float32 volume, accumulated in float64."""
    a = np.asarray(array)
    if a.ndim != 4 or not a.size or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Latent contains nonfinite values or has invalid shape/dtype')
    v = a.reshape(a.shape[0], -1)
    mean = v.mean(axis=1, dtype=np.float64)
    m2 = v.var(axis=1, dtype=np.float64) * v.shape[1]
    return {'schema': 1, 'shape': list(a.shape), 'channel_count': v.shape[1],
            'channel_mean': mean.tolist(), 'channel_m2': m2.tolist(),
            'channel_min': v.min(axis=1).tolist(), 'channel_max': v.max(axis=1).tolist()}


def check_moments(m, shape):
    if m.get('schema') != 1 or m['shape'] != shape or len(shape) != 4 or min(shape) < 1:
        raise ValueError('Invalid latent moment shape/schema')
    if m['channel_count'] != math.prod(shape[1:]):
        raise ValueError('Invalid latent moment count')
    values = np.array([m[k] for k in ('channel_mean','channel_m2','channel_min','channel_max')], dtype=np.float64)
    if values.shape != (4, shape[0]) or not np.isfinite(values).all() or (values[1] < 0).any():
        raise ValueError('Invalid latent moment values')
    if (values[2] > values[0]).any() or (values[0] > values[3]).any():
        raise ValueError('Invalid latent moment bounds')


def combine_moments(records):
    """Merge central moments with sample counts (never average volume stds)."""
    count = 0
    mean = m2 = low = high = shape = None
    samples = 0
    for rec in records:
        current_shape = rec['shape']
        check_moments(rec, current_shape)
        if shape is not None and current_shape != shape:
            raise ValueError('Inconsistent latent shapes')
        shape = current_shape
        n = rec['channel_count']
        bm = np.array(rec['channel_mean'], dtype=np.float64)
        bm2 = np.array(rec['channel_m2'], dtype=np.float64)
        if count == 0:
            mean, m2 = bm, bm2
            low, high = np.array(rec['channel_min']), np.array(rec['channel_max'])
        else:
            delta = bm - mean
            m2 += bm2 + delta * delta * (count * n / (count + n))
            mean += delta * (n / (count + n))
            low = np.minimum(low, rec['channel_min'])
            high = np.maximum(high, rec['channel_max'])
        count += n
        samples += 1
    if not count:
        raise ValueError('Empty latent dataset')
    global_mean = float(mean.mean())
    total = count * len(mean)
    variance = float((m2 + count * (mean - global_mean)**2).sum() / total)
    channel_var = m2 / count
    return {'count': total, 'sample_count': samples, 'per_channel_count': count,
            'global_mean': global_mean, 'global_variance': variance,
            'global_std': math.sqrt(max(variance, 0.)),
            'global_min': float(low.min()), 'global_max': float(high.max()),
            'per_channel_mean_raw': mean.tolist(), 'per_channel_variance_raw': channel_var.tolist(),
            'per_channel_std_raw': np.sqrt(channel_var).tolist(),
            'per_channel_min_raw': low.tolist(), 'per_channel_max_raw': high.tolist(),
            'latent_shape': shape, 'variance_ddof': 0, 'accumulator_dtype': 'float64'}


def streaming_stats(files):
    return combine_moments(moments(np.load(p, allow_pickle=False, mmap_mode='r')) for p in files)


def stamp(path):
    stat = Path(path).stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def save_array(path, array, identity):
    path = Path(path)
    a = np.asarray(array, dtype=np.float32)
    summary = moments(a)  # Small CPU reduction while this volume is already in RAM.
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    atomic_json(path.with_suffix('.json'), {'identity': identity, 'sha256': sha256(path),
                'shape': list(a.shape), 'moments': summary, 'file_stamp': stamp(path)})


def valid_array(path, identity):
    try:
        path = Path(path)
        rec = json.loads(path.with_suffix('.json').read_text())
        if rec['identity'] != identity or rec['sha256'] != sha256(path):
            return False
        a = np.load(path, allow_pickle=False, mmap_mode='r')
        return a.ndim == 4 and list(a.shape) == rec['shape'] and a.dtype == np.float32 and np.isfinite(a).all()
    except (OSError, ValueError, KeyError, EOFError, TypeError):
        return False


def read_record(path, identity, *, verify=False):
    """Fast metadata path; verify old/touched files and backfill missing moments."""
    path = Path(path)
    rec = json.loads(path.with_suffix('.json').read_text())
    if rec['identity'] != identity:
        raise ValueError('Raw latent identity mismatch: ' + str(path))
    current = stamp(path)
    cached = rec.get('file_stamp') == current and 'moments' in rec
    if verify or not cached:
        if sha256(path) != rec['sha256']:
            raise ValueError('Raw latent checksum mismatch: ' + str(path))
        a = np.load(path, allow_pickle=False, mmap_mode='r')
        if list(a.shape) != rec['shape'] or a.dtype != np.float32:
            raise ValueError('Raw latent shape/dtype mismatch: ' + str(path))
        rec['moments'] = moments(a)
        if stamp(path) != current:
            raise ValueError('Raw latent changed during statistics: ' + str(path))
        rec['file_stamp'] = current
        atomic_json(path.with_suffix('.json'), rec)
    check_moments(rec['moments'], rec['shape'])
    return rec


def record_identity(rec):
    # Exclude timestamps: copying/touching identical data must not alter normalization identity.
    return {k: rec[k] for k in ('identity', 'sha256', 'shape', 'moments')}


def finalize(out, *, workers=2, verify_raw=False):
    """One CPU job: aggregate sidecars, publish stats and a raw-only READY marker."""
    out = Path(out).resolve()
    with file_lock(out / 'control/pool.lock') as acquired:
        if not acquired:
            raise RuntimeError('Workers are still active; finalization refused')
        (out / 'READY.json').unlink(missing_ok=True)
        manifest = json.loads((out / 'manifest.json').read_text())
        with stage('verify frozen AE checkpoint'):
            verify_ae_manifest(manifest)
        identity = digest(manifest)
        n = int(manifest['number_of_latents'])
        files = [out / 'raw' / f'latent_{i:08d}.npy' for i in range(n)]
        if n < 1 or workers < 1:
            raise ValueError('Nonempty cache and positive CPU worker count required')
        extras = set((out / 'raw').glob('*.npy')) - set(files)
        if extras:
            raise ValueError('Unexpected raw latent files outside the frozen manifest')
        def read(p):
            try:
                return read_record(p, identity, verify=verify_raw), None
            except (OSError, ValueError, KeyError, EOFError, TypeError) as exc:
                return None, str(exc)
        with stage('aggregate raw latent moment records (CPU)',
                   timeout=float(os.getenv('LATENT_FINALIZE_TIMEOUT_SECONDS', '21600'))):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(read, files))
        bad = [i for i, (_, error) in enumerate(results) if error is not None]
        if bad:
            atomic_json(out / 'control/status.json', {'status': 'incomplete',
                'missing_count': len(bad), 'missing_indices': bad,
                'first_error': results[bad[0]][1]})
            raise RuntimeError(f'{len(bad)} raw latents missing/invalid; rerun the export to resume. '
                               + results[bad[0]][1])
        records = [r for r, _ in results]
        stats = combine_moments(r['moments'] for r in records)
        mode = manifest['normalization']
        if mode not in ('global_sigma', 'zscore'):
            raise ValueError('Unsupported normalization mode: ' + mode)
        sigma = max(stats['global_std'], 1e-6)
        shift = stats['global_mean'] if mode == 'zscore' else 0.
        stats.update(normalization=mode, scale=sigma, shift=shift, manifest_sha256=identity,
                     formula='z_norm = (z_raw - shift) / scale', storage='raw',
                     normalization_at='dataset_read')
        stats['per_channel_mean_normalized'] = [(x - shift) / sigma for x in stats['per_channel_mean_raw']]
        stats['per_channel_variance_normalized'] = [x / sigma**2 for x in stats['per_channel_variance_raw']]
        atomic_json(out / 'stats.json', stats)
        atomic_json(out / 'READY.json', {'schema': 3, 'ready': True, 'storage': 'raw',
            'manifest_sha256': identity, 'stats_sha256': digest(stats), 'raw_count': n,
            'raw_records_sha256': digest([record_identity(r) for r in records])})
        atomic_json(out / 'control/status.json', {'status': 'complete', 'volumes': n,
                                               'storage': 'raw', 'stats': str(out / 'stats.json')})
        log(f'READY: {n} raw latents; stats={out / "stats.json"}; normalization happens when training reads each latent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify-raw', action='store_true', help='Re-read and checksum every raw array (slower)')
    a = parser.parse_args()
    finalize(a.out_root, workers=a.workers, verify_raw=a.verify_raw)


if __name__ == '__main__':
    main()
