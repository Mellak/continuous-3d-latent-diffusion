"""Diffusion inputs: normalize immutable raw latents on each read."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from c3d.data.identity import digest, sha256, verify_ae_manifest
from c3d.data.latent_statistics import check_moments, record_identity, stamp, valid_array


class LatentDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.stats = json.loads((self.root / 'stats.json').read_text())
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        ready = json.loads((self.root / 'READY.json').read_text())
        if not ready.get('ready') or ready.get('schema') not in (2, 3):
            raise ValueError('Cache incomplete or old schema; rebuild it')
        identity = digest(self.manifest)
        if ready['manifest_sha256'] != identity or ready['stats_sha256'] != digest(self.stats):
            raise ValueError('Cache manifest/statistics identity mismatch')
        if self.stats.get('manifest_sha256') != identity:
            raise ValueError('Statistics belong to another cache')
        verify_ae_manifest(self.manifest)
        self.scale = float(self.stats['scale'])
        self.shift = float(self.stats.get('shift', 0.))
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.shift):
            raise ValueError('Invalid normalization scale/shift')
        n = int(self.manifest['number_of_latents'])
        self.on_the_fly = ready['schema'] == 3
        directory = 'raw' if self.on_the_fly else 'normalized'
        self.files = [self.root / directory / f'latent_{i:08d}.npy' for i in range(n)]
        if n < 1 or len(list((self.root / directory).glob('*.npy'))) != n:
            raise ValueError('Cache file count mismatch')
        self.records = []
        self._verified = set()
        if self.on_the_fly:
            if ready.get('storage') != 'raw' or ready['raw_count'] != n:
                raise ValueError('Raw cache storage/count mismatch')
            try:
                for path in self.files:
                    rec = json.loads(path.with_suffix('.json').read_text())
                    if rec['identity'] != identity or rec['shape'] != self.stats['latent_shape']:
                        raise ValueError('Raw cache identity/shape mismatch')
                    check_moments(rec['moments'], rec['shape'])
                    if rec['file_stamp'] != stamp(path):
                        raise ValueError('Raw cache checksum/metadata changed; rerun latent statistics')
                    self.records.append(rec)
            except (OSError, KeyError, EOFError, TypeError) as exc:
                raise ValueError('Raw cache checksum/metadata invalid; rerun export') from exc
            if digest([record_identity(r) for r in self.records]) != ready['raw_records_sha256']:
                raise ValueError('Raw latent records identity mismatch; rerun statistics')
        else:
            # Old pre-normalized caches remain readable, without normalizing twice.
            expected = digest({'manifest': identity, 'stats': digest(self.stats)})
            if (ready['normalized_count'] != n or ready['normalized_identity'] != expected
                    or any(not valid_array(p, expected) for p in self.files)):
                raise ValueError('Cache file checksum/shape/finiteness check failed; rerun export')
        self.identity = digest({'manifest': self.manifest, 'stats': self.stats})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        if self.on_the_fly:
            rec = self.records[i]
            if stamp(path) != rec['file_stamp']:
                raise ValueError('Raw latent changed after dataset initialization: ' + str(path))
            # Full checksum once per latent per loader process, then cheap stamps.
            if i not in self._verified:
                if sha256(path) != rec['sha256']:
                    raise ValueError('Raw latent checksum mismatch: ' + str(path))
                self._verified.add(i)
        a = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if self.on_the_fly:
            if list(a.shape) != rec['shape'] or not np.isfinite(a).all() or stamp(path) != rec['file_stamp']:
                raise ValueError('Raw latent shape/finiteness/identity changed: ' + str(path))
            a -= self.shift
            a /= self.scale
        return torch.from_numpy(a)


def denormalize(z, stats):
    return z * float(stats['scale']) + float(stats.get('shift', 0.0))
