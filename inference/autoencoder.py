#!/usr/bin/env python3
"""Encode and decode one volume with a released CoRe-AE (LIIF or CLIF head).

    python inference/autoencoder.py --model core-ae-liif-ct512 --input ct.nii.gz --out outputs/ae
    python inference/autoencoder.py --model core-ae-clif-ct512 --input ct.nii.gz --scale 2   # decode at 2x resolution

The whole volume is encoded once. The decoder runs its convolutions once on the
latent grid, then the head renders voxels chunk by chunk (LIIF) or tile by tile (CLIF).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from c3d.hub import load_autoencoder, render_kwargs
from inference.common import (autocast, load_input, measure, pick_device, psnr, save_nifti, ssim,
                              three_planes, to_output_units)


@torch.no_grad()
def reconstruct(model_id, input_path, out_dir=None, scale: float = 1.0, device=None,
                precision='bf16', resample=True, preprocessed=False):
    """Return a dict with input, reconstruction, latent, metrics and cost."""
    device = pick_device(device)
    ae, cfg = load_autoencoder(model_id, device)
    x = load_input(input_path, cfg, resample=resample, preprocessed=preprocessed).to(device)
    native = tuple(cfg['native_shape'])
    out_shape = tuple(int(round(n * scale)) for n in native)
    kw = render_kwargs(cfg)
    cost = {}
    with measure(device, cost), autocast(device, precision):
        z, _ = ae.encode(x, sample=False)                  # posterior mean
        recon = ae.decode_latent(z, out_shape, **kw).float()
    result = {'model': model_id, 'input': x.cpu(), 'latent': z.float().cpu(), 'recon': recon.cpu(),
              'output_shape': list(out_shape), 'seconds': cost['seconds'], 'peak_gib': cost.get('peak_gib')}
    if out_shape == native:
        result['psnr'] = psnr(result['recon'], result['input'])
        result['ssim'] = ssim(result['recon'], result['input'])
    if out_dir:
        out = Path(out_dir)
        spacing = cfg.get('voxel_spacing_xyz_mm')
        if spacing and scale != 1:
            spacing = [s / scale for s in spacing]
        save_nifti(to_output_units(result['recon'], cfg), out / f'{model_id}_recon.nii.gz', spacing, cfg.get('array_axes', 'zyx'))
        panels = {'input': result['input'], 'recon': result['recon']} if out_shape == native else {'recon': result['recon']}
        three_planes(panels, out / f'{model_id}_recon.png', model_id, cfg['modality'])
        summary = {k: v for k, v in result.items() if k not in ('input', 'latent', 'recon')}
        (out / f'{model_id}_metrics.json').write_text(json.dumps(summary, indent=2))
    return result


@torch.no_grad()
def render_axial_region(ae, z, native, depth, y0, x0, size, scale=1.0, precision='bf16'):
    """LIIF only: query an axial region directly from coordinates, on a grid `scale` times finer.

    Region in native voxels: rows y0..y0+size, columns x0..x0+size of slice `depth`.
    Returns a [size*scale, size*scale] tensor in [-1, 1]. With scale=1 it equals the
    same region of ``decode_latent``.
    """
    if ae.head_name != 'liif':
        raise ValueError('Region queries are shown for the LIIF head; use decode_latent(..., scale) for CLIF')
    d, h, w = native
    n = int(round(size * scale))
    t = (torch.arange(n, dtype=torch.float32) + 0.5) / scale
    ys, xs = torch.meshgrid(y0 + t, x0 + t, indexing='ij')
    zc = torch.full_like(xs, depth + 0.5)
    coord = torch.stack((xs * 2 / w - 1, ys * 2 / h - 1, zc * 2 / d - 1), -1).reshape(1, -1, 3).to(z.device)
    cell = torch.tensor([2 / (w * scale), 2 / (h * scale), 2 / (d * scale)], device=z.device).view(1, 1, 3).expand(1, n * n, 3)
    with autocast(z.device, precision):
        feat = ae.decode_features(z)
        out = ae.head.query(feat, coord, cell)
    return out.float().reshape(n, n).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='release id (e.g. core-ae-liif-ct512) or local folder')
    p.add_argument('--input', required=True, help='.nii/.nii.gz/.npy/.pt volume')
    p.add_argument('--out', default='outputs/autoencoder')
    p.add_argument('--scale', type=float, default=1.0, help='output resolution relative to the native grid')
    p.add_argument('--device')
    p.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--no-resample', action='store_true', help='do not resample the input to the training spacing')
    p.add_argument('--preprocessed', action='store_true', help='input is already in [-1, 1] on the training grid')
    a = p.parse_args()
    r = reconstruct(a.model, a.input, a.out, a.scale, a.device, a.precision, not a.no_resample, a.preprocessed)
    print(json.dumps({k: v for k, v in r.items() if k not in ('input', 'latent', 'recon')}, indent=2))


if __name__ == '__main__':
    main()
