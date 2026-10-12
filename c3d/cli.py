"""Command line entry point: ``c3d <command>``.

    c3d train-ae       --config configs/ae/ct_liif.yaml   --data /data/ct --out runs/ct_liif
    c3d export-latents --ae runs/ct_liif                  --data /data/ct --out latents/ct_liif
    c3d train-prior    --config configs/prior/ct_jit.yaml --latents latents/ct_liif --out runs/ct_liif_jit
    c3d sample         --prior runs/ct_liif_jit --num 4 --out samples/
    c3d generate       --prior core-ldm-jit-ct512-liif --num 4          (released weights)
    c3d reconstruct    --model core-ae-liif-ct512 --input ct.nii.gz     (released weights)

One GPU: run as is. Several GPUs: ``torchrun --nproc_per_node=N -m c3d.cli <command> ...``.
Running the same training command again with the same --out resumes from checkpoints/last.pt.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import yaml


def _load(path):
    return yaml.safe_load(os.path.expandvars(Path(path).read_text()))


def _set(cfg, assignments):
    """Apply ``--set a.b.c=value`` overrides (value parsed as YAML)."""
    for item in assignments or ():
        key, _, value = item.partition('=')
        if not _:
            raise SystemExit(f'--set expects key=value, got {item!r}')
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def _write_input_config(cfg, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'input_config.yaml'
    text = yaml.safe_dump(cfg, sort_keys=False)
    if os.environ.get('RANK', '0') == '0':
        path.write_text(text)
    return path


def _released_weights(name):
    """Return model.safetensors for a release id/folder, or None for an ordinary checkpoint path."""
    if name is None or (Path(name).is_file()):
        return None
    from c3d.hub import resolve
    return str(resolve(name) / 'model.safetensors')


def train_ae(a):
    cfg = _set(_load(a.config), a.set)
    if a.data:
        cfg['data']['root'] = a.data
    if not cfg['data'].get('root'):
        raise SystemExit('Give the training volumes with --data or data.root in the config')
    init = _released_weights(a.init) or a.init
    if init:
        cfg.setdefault('initialization', {})['checkpoint'] = init
    path = _write_input_config(cfg, Path(a.out))
    from c3d.training.train_ae import train
    return train(str(path), a.out, a.mode, init, True)


def _ae_training_config(ae):
    """(training-format AE config, checkpoint path) for a training run folder or a released AE."""
    run = Path(ae)
    if (run / 'checkpoints' / 'best.pt').exists():
        return _load(run / 'config.yaml'), str(run / 'checkpoints' / 'best.pt')
    from c3d.hub import resolve
    folder = resolve(ae)
    rel = _load(folder / 'config.yaml')
    spacing = rel.get('voxel_spacing_xyz_mm')
    cfg = {'experiment': {'name': rel['id'], 'modality': rel['modality'], 'native_shape': rel['native_shape']},
           'data': {'root': None, 'patterns': None, 'file_list': None,
                    'target_spacing': None if spacing is None else list(spacing),
                    'augment_flip': False, **rel['preprocessing']},
           'model': rel['model'],
           'validation': {'fraction': 0.1, 'seed': 91415}}
    return cfg, str(folder / 'model.safetensors')


def export_latents(a):
    cfg, checkpoint = _ae_training_config(a.ae)
    cfg = _set(cfg, a.set)
    if a.data:
        cfg['data']['root'] = a.data
    if not cfg['data'].get('root'):
        raise SystemExit('Give the training volumes with --data')
    path = _write_input_config(cfg, Path(a.out) / 'control')
    from c3d.data.latent_cache import prepare
    return prepare(str(path), checkpoint, a.out, a.normalization, True)


def train_prior(a):
    cfg = _set(_load(a.config), a.set)
    if a.latents:
        cfg.setdefault('data', {})['latent_root'] = a.latents
    path = _write_input_config(cfg, Path(a.out))
    from c3d.training.train_ldm import train
    return train(str(path), a.out, cfg['data']['latent_root'], True)


def sample(a):
    run = Path(a.prior)
    ckpt = run / 'checkpoints' / f'{a.checkpoint}.pt' if run.is_dir() else run
    from c3d.training.sample_ldm import sample as run_sample
    return run_sample(str(ckpt), a.out, a.num, a.seed, a.steps, a.method)


def generate(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference.generate import generate as run_generate
    for r in run_generate(a.prior, a.num, a.seed, a.steps, a.method, a.out, a.device, a.precision):
        print(f"seed {r['seed']}: {r['seconds']:.1f} s")


def reconstruct(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference.autoencoder import reconstruct as run_reconstruct
    r = run_reconstruct(a.model, a.input, a.out, a.scale, a.device, a.precision, not a.no_resample, a.preprocessed)
    print({k: v for k, v in r.items() if k in ('psnr', 'ssim', 'seconds', 'peak_gib', 'output_shape')})


def main(argv=None):
    p = argparse.ArgumentParser(prog='c3d', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('train-ae', help='train a CoRe-AE (LIIF or CLIF head)')
    s.add_argument('--config', required=True)
    s.add_argument('--data', help='folder of training volumes (overrides data.root)')
    s.add_argument('--out', required=True)
    s.add_argument('--mode', choices=['scratch', 'decoder', 'finetune'],
                   help='scratch: new weights; finetune: start from --init; decoder: also freeze the encoder')
    s.add_argument('--init', help='checkpoint, release id or weight folder to start from')
    s.add_argument('--set', action='append', metavar='KEY=VALUE')
    s.set_defaults(fn=train_ae)

    s = sub.add_parser('export-latents', help='encode the training volumes (posterior mean) and compute statistics')
    s.add_argument('--ae', required=True, help='AE training run folder, release id or weight folder')
    s.add_argument('--data', help='folder of training volumes')
    s.add_argument('--out', required=True)
    s.add_argument('--normalization', choices=['global_sigma', 'zscore'], default='global_sigma')
    s.add_argument('--set', action='append', metavar='KEY=VALUE')
    s.set_defaults(fn=export_latents)

    s = sub.add_parser('train-prior', help='train a DDPM-epsilon or JiT prior on exported latents')
    s.add_argument('--config', required=True)
    s.add_argument('--latents', help='folder written by export-latents (overrides data.latent_root)')
    s.add_argument('--out', required=True)
    s.add_argument('--set', action='append', metavar='KEY=VALUE')
    s.set_defaults(fn=train_prior)

    s = sub.add_parser('sample', help='sample from a prior training run (EMA weights)')
    s.add_argument('--prior', required=True, help='prior run folder or checkpoint file')
    s.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    s.add_argument('--out', required=True)
    s.add_argument('--num', type=int, default=1)
    s.add_argument('--seed', type=int, default=91415)
    s.add_argument('--steps', type=int)
    s.add_argument('--method', choices=['heun', 'euler'])
    s.set_defaults(fn=sample)

    s = sub.add_parser('generate', help='generate with released weights (see inference/generate.py)')
    s.add_argument('--prior', required=True)
    s.add_argument('--num', type=int, default=1)
    s.add_argument('--seed', type=int, default=91415)
    s.add_argument('--steps', type=int)
    s.add_argument('--method', choices=['heun', 'euler'])
    s.add_argument('--out', default='outputs/generate')
    s.add_argument('--device')
    s.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'])
    s.set_defaults(fn=generate)

    s = sub.add_parser('reconstruct', help='encode/decode a volume with released weights (see inference/autoencoder.py)')
    s.add_argument('--model', required=True)
    s.add_argument('--input', required=True)
    s.add_argument('--out', default='outputs/autoencoder')
    s.add_argument('--scale', type=float, default=1.0)
    s.add_argument('--device')
    s.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    s.add_argument('--no-resample', action='store_true')
    s.add_argument('--preprocessed', action='store_true')
    s.set_defaults(fn=reconstruct)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    main()
