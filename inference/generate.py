#!/usr/bin/env python3
"""Generate full volumes with a latent prior and decode them with its CoRe-AE.

    python inference/generate.py --prior core-ldm-jit-ct512-liif  --num 4 --seed 91415
    python inference/generate.py --prior core-ldm-ddpm-ct512-clif --num 1
    python inference/generate.py --prior core-ldm-jit-mri256-clif --num 4 --steps 50

JiT priors are sampled with Heun steps (default 50). DDPM-epsilon priors use
their trained 1000-step ancestral schedule. Each prior is paired with the
autoencoder whose latents it was trained on (``autoencoder`` in its config).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from c3d.hub import load_autoencoder, load_prior, render_kwargs
from c3d.diffusion.samplers.heun import sample as jit_sample
from c3d.diffusion.ddpm import make_schedule, sample as ddpm_sample
from inference.common import autocast, measure, pick_device, save_nifti, three_planes, to_output_units

PRIOR_ALIASES = {f'{kind}-{head}': f'core-ldm-{kind.split("-")[1]}-{kind.split("-")[0]}{"512" if kind.startswith("ct") else "256"}-{head}'
                 for kind in ('ct-jit', 'ct-ddpm', 'mri-jit') for head in ('liif', 'clif')}


def denormalize(z, stats):
    return z * float(stats['scale']) + float(stats.get('shift', 0.0))


@torch.no_grad()
def sample_latent(prior, cfg, stats, seed, steps=None, method=None, device='cuda', precision=None):
    """One normalised latent [1, C, d, h, w]."""
    device = torch.device(device)
    shape = (1, *[int(v) for v in stats['latent_shape']])
    obj = cfg['objective']
    precision = precision or cfg['sampling']['precision']
    with autocast(device, precision):
        if obj['name'] == 'jit_x0v':
            steps = int(steps or cfg['sampling']['steps'])
            return jit_sample(prior, shape, obj, steps, method or cfg['sampling']['method'], seed, device).float()
        if steps is not None or method is not None:
            raise ValueError('DDPM priors use their trained schedule; --steps/--method apply to JiT only')
        sch = make_schedule(int(obj.get('num_steps', 1000)), float(obj.get('beta_start', 1e-4)),
                            float(obj.get('beta_end', 2e-2)), obj.get('schedule', 'linear'))
        return ddpm_sample(prior, shape, sch, 'eps', seed, device).float()


def generate(prior_id, num=1, seed=91415, steps=None, method=None, out_dir=None, device=None,
             precision=None, save_latents=True):
    """Generate `num` volumes (seeds seed, seed+1, ...). Returns a list of result dicts."""
    prior_id = PRIOR_ALIASES.get(prior_id, prior_id)
    device = pick_device(device)
    prior, cfg, stats = load_prior(prior_id, device)
    ae, ae_cfg = load_autoencoder(cfg['autoencoder'], device)
    native = tuple(ae_cfg['native_shape'])
    kw = render_kwargs(ae_cfg)
    precision = precision or cfg['sampling']['precision']
    results = []
    for i in range(int(num)):
        cost = {}
        with measure(device, cost):
            zn = sample_latent(prior, cfg, stats, seed + i, steps, method, device, precision)
            z = denormalize(zn, stats)
            with torch.no_grad(), autocast(device, precision):
                vol = ae.decode_latent(z, native, **kw).float()
        r = {'prior': prior_id, 'autoencoder': cfg['autoencoder'], 'modality': ae_cfg['modality'], 'seed': seed + i,
             'volume': vol.cpu(), 'latent': z.cpu(), 'seconds': cost['seconds'], 'peak_gib': cost.get('peak_gib')}
        if out_dir:
            out = Path(out_dir)
            stem = f'{prior_id}_seed{seed + i}'
            save_nifti(to_output_units(r['volume'], ae_cfg), out / f'{stem}.nii.gz', ae_cfg.get('voxel_spacing_xyz_mm'), ae_cfg.get('array_axes', 'zyx'))
            if save_latents:
                np.save(out / f'{stem}_latent.npy', r['latent'][0].numpy())
            three_planes({prior_id: r['volume']}, out / f'{stem}.png', f'seed {seed + i}', ae_cfg['modality'])
        results.append(r)
        del zn, z, vol
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if out_dir:
        meta = [{k: v for k, v in r.items() if k not in ('volume', 'latent')} for r in results]
        (Path(out_dir) / f'{prior_id}_generation.json').write_text(json.dumps(meta, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prior', required=True,
                   help='release id (e.g. core-ldm-jit-ct512-liif), short form (ct-jit-liif) or local folder')
    p.add_argument('--num', type=int, default=1)
    p.add_argument('--seed', type=int, default=91415)
    p.add_argument('--steps', type=int, help='JiT only (default: 50)')
    p.add_argument('--method', choices=['heun', 'euler'], help='JiT only (default: heun)')
    p.add_argument('--out', default='outputs/generate')
    p.add_argument('--device')
    p.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'])
    a = p.parse_args()
    for r in generate(a.prior, a.num, a.seed, a.steps, a.method, a.out, a.device, a.precision):
        print(f"seed {r['seed']}: {r['seconds']:.1f} s, peak {r['peak_gib'] or 0:.2f} GiB")


if __name__ == '__main__':
    main()
