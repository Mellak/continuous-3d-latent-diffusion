"""EMA validation with fixed per-volume noise and timesteps on every visit."""
from __future__ import annotations

import math
import torch
import torch.distributed as dist

from c3d.diffusion.ddpm import extract
from c3d.diffusion.jit_x0v import objective as jit_objective
from c3d.training.resume import capture_rng, restore_rng
from c3d.training.runtime import autocast, coordinated_stop


@torch.no_grad()
def validate(model, ema, dataset, config, schedule, device, rank, world, run):
    vc = config.get('validation', {})
    count = min(len(dataset), int(vc.get('max_volumes', 4)))
    repeats = int(vc.get('noise_draws', 2))
    if count < 1 or repeats < 1:
        raise ValueError('Validation needs positive max_volumes and noise_draws')
    obj = config['objective']
    saved_rng = capture_rng()
    was_training = model.training
    total = torch.zeros(2, device=device, dtype=torch.float64)
    try:
        with ema.applied(model):
            model.eval()
            # Equal numbers of collectives even when the holdout is smaller
            # than world_size. No padding/duplicate validation samples.
            for offset in range(math.ceil(count/world)):
                if coordinated_stop(run, device):
                    return None
                position = offset * world + rank
                x = dataset[position].unsqueeze(0).to(device) if position < count else None
                for draw in range(repeats):
                    if coordinated_stop(run, device):
                        return None
                    if x is None:
                        continue
                    source_index = int(dataset.indices[position])
                    seed = int(vc.get('seed', 91415)) + source_index * 1000003 + draw * 9176
                    generator = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn(x.shape, generator=generator, device=device, dtype=torch.float32)
                    with autocast(device, device.type == 'cuda', config['training'].get('precision', 'bf16')):
                        if obj['name'] == 'jit_x0v':
                            t = (torch.randn(1, generator=generator, device=device) * float(obj['p_std'])
                                 + float(obj['p_mean'])).sigmoid()
                            loss, _ = jit_objective(model, x, obj, t=t, noise=noise)
                        else:
                            t = torch.randint(schedule.num_steps, (1,), generator=generator, device=device)
                            xt = extract(schedule.sqrt_alphas_cumprod, t, x.shape) * x + extract(schedule.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
                            pred = model(xt, t)
                            target = noise if obj['name'] == 'ddpm_eps' else x
                            loss = (pred.float() - target.float()).square().mean()
                    total[0] += loss.detach().double()
                    total[1] += 1
            if dist.is_initialized():
                dist.all_reduce(total)
            value = float(total[0] / total[1])
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite validation loss')
            return value
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
