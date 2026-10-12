"""Bit-identity check between this package and the original research code.

Run only when the original project is available:

    C3D_REFERENCE_ROOT=/path/to/Continuous3D_AE_LDM \
    C3D_REFERENCE_AE=/path/to/ae_checkpoint.pt \
    C3D_REFERENCE_PRIOR=/path/to/prior_checkpoint.pt \
    pytest tests/test_equivalence.py -q

Both code bases are imported side by side (the reference uses top-level
packages ``models``/``diffusion``; this repository uses ``c3d``).
"""
from __future__ import annotations
import os
import sys
import pytest
import torch

ROOT = os.environ.get('C3D_REFERENCE_ROOT')
AE = os.environ.get('C3D_REFERENCE_AE')
PRIOR = os.environ.get('C3D_REFERENCE_PRIOR')
pytestmark = pytest.mark.skipif(not ROOT, reason='C3D_REFERENCE_ROOT not set')


def _reference():
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    import models.ae.autoencoder_3d as old_ae
    import models.ldm.model as old_ldm
    return old_ae, old_ldm


@pytest.mark.skipif(not AE, reason='C3D_REFERENCE_AE not set')
def test_autoencoder_identical():
    old_ae, _ = _reference()
    from c3d.models.ae.autoencoder_3d import load_complete_model
    meta = torch.load(AE, map_location='cpu', weights_only=False)
    cfg = meta['config']
    old = old_ae.load_complete_model(cfg, AE, 'cpu')
    new = load_complete_model(cfg, AE, 'cpu')
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32, 32) * 2 - 1
    z = torch.randn(1, cfg['model']['encoder']['latent_channels'], 8, 8, 8)
    hc = cfg['model']['head']['config']
    kw = dict(chunk=4096, tile_size=(16, 16, 16), halo=int(hc.get('halo', 6)))
    with torch.no_grad():
        assert torch.equal(old.encode(x, sample=False)[0], new.encode(x, sample=False)[0])
        for shape in ((32, 32, 32), (48, 40, 32)):          # native and another output size
            a = old.decode_latent(z, shape, **kw)
            b = new.decode_latent(z, shape, **kw)
            assert torch.equal(a, b), shape


@pytest.mark.skipif(not PRIOR, reason='C3D_REFERENCE_PRIOR not set')
def test_prior_identical():
    _, old_ldm = _reference()
    from c3d.models.ldm.model import build_ldm
    c = torch.load(PRIOR, map_location='cpu', weights_only=False)
    cfg, stats = c['config'], c['latent_stats']
    mc = dict(cfg['model'])
    nets = []
    for build in (old_ldm.build_ldm, build_ldm):
        m = build(dict(mc), cfg['objective'], stats).eval()
        m.load_state_dict(c['model'], strict=True)
        nets.append(m)
    torch.manual_seed(0)
    channels = int(stats['latent_shape'][0])
    z = torch.randn(1, channels, 32, 32, 32)
    t = torch.full((1,), 0.37) if cfg['objective']['name'] == 'jit_x0v' else torch.full((1,), 517, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(nets[0](z, t), nets[1](z, t))
