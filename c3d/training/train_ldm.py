#!/usr/bin/env python3
from __future__ import annotations
import argparse,time,traceback,os
from pathlib import Path
import torch,torch.distributed as dist,yaml
from c3d.data.latent_dataset import LatentDataset,denormalize
from c3d.models.ldm.model import build_ldm
from c3d.models.ldm.ema import EMA
from c3d.diffusion.jit_x0v import objective as jit_objective
from c3d.diffusion.samplers.heun import sample as jit_sample
from c3d.diffusion.ddpm import make_schedule,objective as ddpm_objective,sample as ddpm_sample
from c3d.training.runtime import setup_distributed,wrap_ddp,is_main,barrier,cleanup,autocast,append_csv,atomic_json,should_stop,system_info,parameter_count,reduce_metrics
from c3d.training.checkpoint import save_ldm,find_last

from c3d.data.identity import digest,verify_ae_manifest
from c3d.models.ae.autoencoder_3d import load_complete_model
from c3d.training.resume import loader_for_epoch,gather_rng,restore_training_rng,capture_rng,restore_rng
from c3d.training.runtime import coordinated_stop,seconds_left,scaler
from c3d.visualization.ae_preview import render_planes,save_planes

def load_yaml(p):return yaml.safe_load(os.path.expandvars(Path(p).read_text()))
def _latent_shape(ds):return tuple(ds.stats['latent_shape'])
def _load_ae_for_preview(manifest,device):
    cfg=verify_ae_manifest(manifest)
    return load_complete_model(cfg,manifest['ae_checkpoint'],device),cfg

def _preview(model_raw,ema,cfg,ds,run,step,device):
    states=capture_rng();was=model_raw.training;obj=cfg['objective'];tr=cfg['training'];shape=(1,*_latent_shape(ds));ae=None
    try:
        with ema.applied(model_raw),torch.no_grad(),autocast(device,device.type=='cuda',tr.get('precision','bf16')):
            model_raw.eval()
            if obj['name']=='jit_x0v':z=jit_sample(model_raw,shape,obj,int(tr.get('sample_steps',50)),tr.get('sampling_method','heun'),int(tr.get('sample_seed',91415)),device,stop=lambda:should_stop(run))
            else:
                sch=make_schedule(int(obj.get('num_steps',1000)),float(obj.get('beta_start',1e-4)),float(obj.get('beta_end',2e-2)),obj.get('schedule','linear'))
                z=ddpm_sample(model_raw,shape,sch,'x0' if obj['name']=='ddpm_x0' else 'eps',int(tr.get('sample_seed',91415)),device,stop=lambda:should_stop(run))
        if z is None or should_stop(run):return False
        zraw=denormalize(z.float(),ds.stats);ae,ae_cfg=_load_ae_for_preview(ds.manifest,device);hc=ae_cfg['model']['head']['config'];native=tuple(ds.manifest['native_shape'])
        with torch.no_grad(),autocast(device,device.type=='cuda',tr.get('precision','bf16')):
            features=ae.decode_features(zraw)
            planes=render_planes(ae,features,native,int(hc.get('render_chunk',65536)),tuple(hc.get('render_tile',[32,64,64])),int(hc.get('halo',8)),lambda:should_stop(run))
        if planes is None or should_stop(run):return False
        target=Path(run)/'images'/f'step_{step:09d}_generated.png'
        temporary=target.with_name(target.stem+'.tmp.png')
        save_planes(planes,temporary,title=f'CLDM | EMA step {step} | {obj["name"]}')
        temporary.replace(target)
        return True
    finally:
        model_raw.train(was);restore_rng(states)
        del ae
        if device.type=='cuda':torch.cuda.empty_cache()


from c3d.training.ldm_data import prepare_data
from c3d.training.ldm_validation import validate


def _trim_metrics(path, step, validation_step):
    """Discard work newer than the durable checkpoint after an abnormal exit."""
    import csv
    if not path.exists():
        return
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [r for r in reader if int(r['step']) <= step and
                (r.get('event') != 'validation' or int(r['step']) <= validation_step)]
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def train(config_path, run_dir, latent_root=None, resume=True):
    cfg = load_yaml(config_path)
    tr, obj = cfg['training'], cfg['objective']
    latent_root = latent_root or cfg['data']['latent_root']
    run = Path(run_dir)
    distributed, rank, local, world, device = setup_distributed(int(tr.get('seed', 0)))
    try:
        precision = tr.get('precision', 'bf16')
        if precision not in ('bf16', 'fp16', 'fp32'):
            raise ValueError('training.precision must be bf16, fp16 or fp32')
        if device.type == 'cuda' and precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This GPU does not support BF16; request BF16-capable 48 GB GPUs')
        expected = tr.get('required_gpus')
        if expected is not None and world != int(expected):
            raise RuntimeError(f'This config requires exactly {expected} ranks (training.required_gpus); '
                               f'launch with torchrun --nproc_per_node={expected} or set training.required_gpus: null')
        for field in ('log_every', 'checkpoint_every'):
            if int(tr.get(field, 100 if field == 'log_every' else 1000)) < 1:
                raise ValueError(field + ' must be positive')
        if is_main(rank):
            for directory in ('checkpoints', 'images', 'control', 'logs'):
                (run/directory).mkdir(parents=True, exist_ok=True)
            print(f'[CLDM] Opening latent cache: {latent_root}', flush=True)
            atomic_json(run/'control/status.json', {'status': 'starting'})
        barrier()
        ds = LatentDataset(latent_root)
        train_ds, val_ds, split = prepare_data(ds, cfg)
        if is_main(rank):
            print(f'[CLDM] train={len(train_ds)}, holdout={len(val_ds)}, shape={ds.stats["latent_shape"]}; '
                  f'normalization=(z - {ds.shift:.8g}) / {ds.scale:.8g}', flush=True)
        model_cfg = dict(cfg['model'])
        model_raw = build_ldm(model_cfg, obj, ds.stats).to(device).train()
        ddp = wrap_ddp(model_raw, device, distributed)
        opt = torch.optim.AdamW(model_raw.parameters(), lr=float(tr.get('learning_rate', 1e-4)),
                               weight_decay=float(tr.get('weight_decay', 0)), foreach=False)
        ema = EMA(model_raw, float(tr.get('ema_decay', .9999)))
        amp_enabled = device.type == 'cuda' and precision in ('bf16', 'fp16')
        sc = scaler(device, amp_enabled, precision)
        step = epoch = cursor = last_validation_step = last_preview_step = 0
        best_val = None
        last = find_last(run) if resume else None
        if last:
            c = torch.load(last, map_location='cpu', weights_only=False)
            if c.get('schema') != 3:
                raise ValueError('Use a fresh CLDM run directory: old checkpoints lack the validation/split state')
            if c.get('config_identity') != digest(cfg) or c.get('latent_identity') != ds.identity or c.get('split_identity') != digest(split):
                raise ValueError('CLDM resume config/latent/split identity changed; start a new run')
            if int(c['world_size']) != world:
                raise ValueError('Resume requires the original GPU count')
            model_raw.load_state_dict(c['model'], strict=True)
            opt.load_state_dict(c['optimizer']); ema.load_state_dict(c['ema']); sc.load_state_dict(c['scaler'])
            step, epoch, cursor = c['step'], c['epoch'], c['cursor']
            best_val = c['best_validation_loss']
            last_validation_step = c['last_validation_step']
            last_preview_step = c['last_preview_step']
            restore_training_rng(c, rank, world, int(tr.get('seed', 0)))
            del c
            if is_main(rank):
                _trim_metrics(run/'metrics.csv', step, last_validation_step)
                for path in (run/'images').glob('step_*_generated.png'):
                    if int(path.name.split('_')[1]) > step:
                        path.unlink()
        elif is_main(rank) and (run/'metrics.csv').exists():
            raise ValueError('Run has metrics but no resumable checkpoint; choose a fresh run directory')
        if is_main(rank):
            (run/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            atomic_json(run/'latent_stats.json', ds.stats)
            atomic_json(run/'split.json', split)
            atomic_json(run/'hardware.json', system_info())
            atomic_json(run/'ldm_identity.json', {'latent_root': str(Path(latent_root).resolve()),
                        'latent_identity': ds.identity, 'split_identity': digest(split), 'objective': obj['name']})
            print(f'[CLDM] {obj["name"]}, {precision}, {world} rank(s), '
                  f'global_batch={int(tr.get("batch_per_gpu",1))*world}, parameters={parameter_count(model_raw):,}; '
                  f'{"resuming" if last else "starting"} step {step}/{tr["max_steps"]}', flush=True)
        sch = (make_schedule(int(obj.get('num_steps', 1000)), float(obj.get('beta_start', 1e-4)),
                             float(obj.get('beta_end', 2e-2)), obj.get('schedule', 'linear'))
               if obj['name'].startswith('ddpm_') else None)

        def checkpoint(is_best=False):
            rng = gather_rng()
            if is_main(rank):
                payload = {'schema': 3, 'model': model_raw.state_dict(), 'optimizer': opt.state_dict(),
                    'ema': ema.state_dict(), 'scaler': sc.state_dict(), 'step': step, 'epoch': epoch,
                    'cursor': cursor, 'rng_states': rng, 'world_size': world, 'config': cfg,
                    'config_identity': digest(cfg), 'latent_identity': ds.identity,
                    'latent_manifest': ds.manifest, 'latent_stats': ds.stats, 'split': split,
                    'split_identity': digest(split), 'batch_policy': 'per_gpu',
                    'best_validation_loss': best_val, 'last_validation_step': last_validation_step,
                    'last_preview_step': last_preview_step}
                save_ldm(run, payload, is_best)
                print(f'[CLDM] Saved last.pt at step {step}' + (' and best.pt' if is_best else ''), flush=True)
            barrier()

        def metrics(event, **values):
            if not is_main(rank):
                return
            row = {'event': event, 'step': step, 'epoch': epoch, 'loss': '', 'val_loss': '', 't_mean': '',
                   'grad_norm': '', 'step_seconds': '', 'world_size': world,
                   'global_batch': int(tr.get('batch_per_gpu', 1))*world,
                   'peak_allocated_gib': '', 'peak_reserved_gib': ''}
            row.update(values)
            if device.type == 'cuda':
                row.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            append_csv(run/'metrics.csv', row)

        def events():
            """Replay due validation/preview after a checkpointed interruption."""
            nonlocal best_val, last_validation_step, last_preview_step
            ve = int(cfg.get('validation', {}).get('every_steps', 5000))
            pe = int(tr.get('sample_every', 5000))
            validation_due = len(val_ds) > 0 and ve > 0 and step > 0 and step % ve == 0 and last_validation_step < step
            preview_due = pe > 0 and step > 0 and step % pe == 0 and last_preview_step < step
            if not (validation_due or preview_due):
                return False
            reserve = float(tr.get('preview_reserve_seconds', 600))
            postpone = torch.tensor(int(seconds_left() <= reserve or should_stop(run)), device=device)
            if distributed:
                dist.all_reduce(postpone, op=dist.ReduceOp.MAX)
            if postpone.item():
                return True  # Save/pause now; carry this event into the next allocation.
            checkpoint()
            if validation_due:
                if is_main(rank):print(f'[CLDM] EMA validation at step {step}', flush=True)
                val = validate(model_raw, ema, val_ds, cfg, sch, device, rank, world, run)
                if val is None:
                    return True
                better = best_val is None or val < best_val
                if better:best_val = val
                last_validation_step = step
                metrics('validation', val_loss=val)
                if is_main(rank):print(f'[CLDM] validation={val:.7g}, best={best_val:.7g}', flush=True)
                checkpoint(better)
            if preview_due:
                result = [None, False]
                if is_main(rank):
                    try:
                        print(f'[CLDM] Generating EMA preview at step {step}', flush=True)
                        result[1] = bool(_preview(model_raw, ema, cfg, ds, run, step, device))
                    except Exception:
                        result[0] = traceback.format_exc()
                if distributed:dist.broadcast_object_list(result, src=0)
                if result[0]:raise RuntimeError('Preview failed: ' + result[0])
                if not result[1]:return True
                last_preview_step = step
                checkpoint()
            return False

        stopped = events()
        while not stopped and step < int(tr['max_steps']):
            if coordinated_stop(run, device):stopped = True; break
            loader, batch = loader_for_epoch(train_ds, tr, rank, world, epoch, cursor)
            for x in loader:
                if coordinated_stop(run, device):stopped = True; break
                if step >= int(tr['max_steps']):break
                t0 = time.time()
                x = x.to(device, non_blocking=True); opt.zero_grad(set_to_none=True)
                with autocast(device, amp_enabled, precision):
                    if obj['name'] == 'jit_x0v':loss, details = jit_objective(ddp, x, obj)
                    elif obj['name'] == 'ddpm_x0':loss, details = ddpm_objective(ddp, x, sch, 'x0')
                    elif obj['name'] == 'ddpm_eps':loss, details = ddpm_objective(ddp, x, sch, 'eps')
                    else:raise ValueError(obj['name'])
                finite = torch.tensor(int(torch.isfinite(loss)), device=device)
                if distributed:dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():raise FloatingPointError(f'Nonfinite CLDM loss at step {step}')
                sc.scale(loss).backward(); sc.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model_raw.parameters(), float(tr.get('grad_clip', 1)), error_if_nonfinite=True)
                sc.step(opt); sc.update(); ema.update(model_raw)
                opt.zero_grad(set_to_none=True)
                step += 1; cursor = min(len(train_ds), cursor + batch*world)
                if step % int(tr.get('log_every', 100)) == 0:
                    av = reduce_metrics({'loss': float(loss.detach()), 't_mean': float(details['t_mean'])}, 1, device)
                    metrics('train', **av, grad_norm=float(gn), step_seconds=time.time()-t0)
                    if is_main(rank):
                        print(f'[CLDM] step={step} loss={av["loss"]:.7g}', flush=True)
                        atomic_json(run/'control/status.json', {'status': 'running', 'step': step, 'world_size': world})
                if step % int(tr.get('checkpoint_every', 1000)) == 0:checkpoint()
                if events():stopped = True; break
            if not stopped and cursor == len(train_ds):epoch += 1; cursor = 0
        checkpoint()
        if is_main(rank):
            status = 'paused' if stopped else 'complete'
            atomic_json(run/'control/status.json', {'status': status, 'step': step, 'world_size': world})
            print(f'[CLDM] {status} at step {step}; resume checkpoint: {run/"checkpoints/last.pt"}', flush=True)
    except Exception:
        if is_main(rank):atomic_json(run/'control/status.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise
    finally:
        cleanup()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--run-dir', required=True)
    p.add_argument('--latent-root'); p.add_argument('--no-resume', action='store_true')
    a = p.parse_args()
    train(a.config, a.run_dir, a.latent_root, not a.no_resume)


if __name__ == '__main__':
    main()
