"""Load released CoRe-AE autoencoders and latent priors.

A model is given either as a local folder (``config.yaml`` + ``model.safetensors``)
or as a release id such as ``core-ae-liif-ct512``. An id is looked up in
``<repo>/weights/<id>/`` (or ``$C3D_WEIGHTS/<id>/``). The autoencoders are stored in
the repository. A prior's weights (331 MB) are downloaded on first use from the GitHub
release asset ``<id>.safetensors`` into ``weights/<id>/``; if that fails, the model is
downloaded from the Hugging Face repository ``C3D_HF_REPO``.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import torch
import yaml

HF_REPO = os.environ.get('C3D_HF_REPO', 'Mellak/continuous-3d-latent-diffusion')
REPO_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'      # <repo>/weights/<id>/
GITHUB_RELEASE = os.environ.get(
    'C3D_RELEASE_URL', 'https://github.com/Mellak/continuous-3d-latent-diffusion/releases/download/v1.0')


def _download_release_asset(name: str, folder: Path) -> bool:
    """Fetch <name>.safetensors from the GitHub release into folder/model.safetensors."""
    import urllib.request
    target = folder / 'model.safetensors'
    partial = folder / 'model.safetensors.part'
    try:
        print(f'Downloading {name} weights from {GITHUB_RELEASE} ...', flush=True)
        urllib.request.urlretrieve(f'{GITHUB_RELEASE}/{name}.safetensors', partial)
        partial.replace(target)
        return True
    except Exception as exc:                       # no network, no release yet, ...
        partial.unlink(missing_ok=True)
        print(f'GitHub release download failed ({exc}); trying Hugging Face.', flush=True)
        return False

AUTOENCODERS = ('core-ae-liif-ct512', 'core-ae-clif-ct512', 'core-ae-liif-mri256', 'core-ae-clif-mri256')
PRIORS = ('core-ldm-ddpm-ct512-liif', 'core-ldm-ddpm-ct512-clif', 'core-ldm-jit-ct512-liif',
          'core-ldm-jit-ct512-clif', 'core-ldm-jit-mri256-liif', 'core-ldm-jit-mri256-clif')


def resolve(name_or_path: str | Path) -> Path:
    """Return a local folder for a path or a release id."""
    path = Path(name_or_path).expanduser()
    if (path / 'config.yaml').exists():
        return path
    name = str(name_or_path)
    local = Path(os.environ.get('C3D_WEIGHTS', REPO_WEIGHTS)) / name
    if (local / 'model.safetensors').exists():
        return local
    if (local / 'config.yaml').exists() and _download_release_asset(name, local):
        return local
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=HF_REPO, allow_patterns=[f'{name}/*'])
    folder = Path(root) / name
    if not (folder / 'config.yaml').exists():
        raise FileNotFoundError(f'{name} was not found in {HF_REPO}')
    return folder


def _state(folder: Path):
    from safetensors.torch import load_file
    return load_file(str(folder / 'model.safetensors'))


def read_config(name_or_path) -> dict:
    return yaml.safe_load((resolve(name_or_path) / 'config.yaml').read_text())


def load_autoencoder(name_or_path, device='cpu'):
    """Return (CoReAE in eval mode, release config)."""
    from c3d.models.ae.autoencoder_3d import build_autoencoder
    folder = resolve(name_or_path)
    cfg = yaml.safe_load((folder / 'config.yaml').read_text())
    model = build_autoencoder({'model': cfg['model']})
    model.load_state_dict(_state(folder), strict=True)
    return model.to(device).eval(), cfg


def load_prior(name_or_path, device='cpu'):
    """Return (denoiser in eval mode with EMA weights, release config, latent statistics)."""
    from c3d.models.ldm.model import build_ldm
    folder = resolve(name_or_path)
    cfg = yaml.safe_load((folder / 'config.yaml').read_text())
    stats = json.loads((folder / 'latent_stats.json').read_text())
    model = build_ldm(dict(cfg['model']), cfg['objective'], stats)
    model.load_state_dict(_state(folder), strict=True)
    return model.to(device).eval(), cfg, stats


def render_kwargs(ae_cfg: dict) -> dict:
    """Chunk/tile settings saved with the autoencoder (they bound memory, not the result)."""
    hc = ae_cfg['model']['head'].get('config', {})
    return dict(chunk=int(hc.get('render_chunk', 300_000)),
                tile_size=tuple(hc.get('render_tile', (32, 64, 64))),
                halo=int(hc.get('halo', 6)))
