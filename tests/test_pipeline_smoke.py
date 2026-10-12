"""Tiny end-to-end run of the training pipeline through the `c3d` command.

train-ae (LIIF and CLIF) -> export-latents -> train-prior (JiT and DDPM-eps) -> sample,
on 16^3 synthetic CT volumes with very small networks. Runs on CPU in a few minutes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from c3d.cli import main

ROOT = Path(__file__).resolve().parents[1]

AE_SET = ['experiment.native_shape=[16,16,16]', 'model.encoder.base_channels=2',
          'model.feature_decoder.stage_channels=[8]', 'model.feature_decoder.stage_blocks=[1]',
          'model.feature_decoder.stage_strides=[1]', 'model.head.config.render_chunk=73',
          'training.epochs=1', 'training.num_workers=0', 'training.train_resolutions=[[16,16,16]]',
          'training.checkpoint_every_steps=1', 'training.losses.perceptual_weight=0',
          'training.losses.cov_weight=0', 'validation.max_volumes=1', 'validation.patch_size=[4,4,4]',
          'validation.patches_per_volume=1']
HEAD_SET = {
    'liif': ['model.head.config.mlp_hidden=[8]', 'training.coords_per_batch=16'],
    'clif': ['model.head.config.hidden_channels=8', 'model.head.config.residual_blocks=1',
             'model.head.config.training_patch=[8,8,8]', 'model.head.config.render_tile=[4,4,4]',
             'model.head.config.halo=4', 'training.losses.gan_weight=0.1',
             'discriminator={base_channels: 4, layers: 1, max_channels: 8}'],
}
PRIOR_SET = ['model.model_channels=8', 'model.channel_mult=[1,2]', 'model.num_res_blocks=1',
             'model.attention_resolutions=[]', 'model.num_heads=1', 'training.max_steps=2',
             'training.workers_per_rank=0', 'training.sample_every=1', 'training.sample_steps=2',
             'training.checkpoint_every=1', 'training.log_every=1', 'validation.every_steps=1',
             'validation.max_volumes=1', 'validation.noise_draws=1']


def _sets(values):
    out = []
    for v in values:
        out += ['--set', v]
    return out


@pytest.mark.parametrize('head', ['liif', 'clif'])
def test_pipeline(tmp_path, head):
    data = tmp_path / 'data'
    data.mkdir()
    for i in range(5):
        np.save(data / f'ct_{i}.npy', np.random.default_rng(i).uniform(-900, 900, (16, 16, 16)).astype('float32'))
    ae = tmp_path / 'ae'
    main(['train-ae', '--config', str(ROOT / f'configs/ae/ct_{head}.yaml'), '--data', str(data),
          '--out', str(ae), '--mode', 'scratch', *_sets(AE_SET + HEAD_SET[head])])
    assert (ae / 'checkpoints' / 'best.pt').exists()
    latents = tmp_path / 'latents'
    main(['export-latents', '--ae', str(ae), '--data', str(data), '--out', str(latents)])
    assert (latents / 'READY.json').exists()
    for kind in ('jit', 'ddpm'):
        prior = tmp_path / f'prior_{kind}'
        extra = ['objective.num_steps=4'] if kind == 'ddpm' else []
        main(['train-prior', '--config', str(ROOT / f'configs/prior/ct_{kind}_{head}.yaml'),
              '--latents', str(latents), '--out', str(prior), *_sets(PRIOR_SET + extra)])
        assert (prior / 'checkpoints' / 'last.pt').exists()
        main(['sample', '--prior', str(prior), '--checkpoint', 'last', '--out', str(tmp_path / f'samples_{kind}')])
        assert (tmp_path / f'samples_{kind}' / 'sample_0000_volume.npy').exists()
