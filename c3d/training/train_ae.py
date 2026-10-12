#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,math,time,os,traceback,random
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from c3d.models.ae.autoencoder_3d import build_autoencoder,initialize_from_checkpoint,set_training_mode
from c3d.models.ae.gaussian import DiagonalGaussian3D
from c3d.models.checkpointing import loss_checkpoint
from c3d.models.discriminator.patchgan_3d import PatchDiscriminator3D
from c3d.losses import l1_loss,stable_covariance_loss,hinge_discriminator_loss,hinge_generator_loss,adaptive_gan_weight,ssim3d_loss
from c3d.losses.perceptual_3d import MONAIPerceptualLoss3D
from c3d.data.medical_volume import MedicalVolumeDataset3D,parse_float3
from c3d.data.coordinate_sampling import sample_liif_coordinates,cells_for_resolution,sample_volume_points,random_dense_patch_coordinates,sample_dense_patch,crop_halo,choose_resolution
from c3d.training.runtime import setup_distributed,wrap_ddp,unwrap,is_main,barrier,cleanup,autocast,scaler,set_requires_grad,append_csv,atomic_json,should_stop,reduce_metrics,system_info,parameter_count
from c3d.training.checkpoint import save_ae,find_last
from c3d.visualization.ae_preview import save_ae_preview


from c3d.data.identity import digest
from c3d.data.splits import split_datasets
from c3d.training.resume import loader_for_epoch,gather_rng,restore_training_rng,capture_rng,restore_rng
from c3d.training.runtime import coordinated_stop,seconds_left
from c3d.training.validation import validate
from c3d.training.progress import log,trim_csv
from c3d.training.startup import StartupMonitor
import torch.distributed as dist

def load_yaml(path):
    return yaml.safe_load(os.path.expandvars(Path(path).read_text()))

def make_dataset(cfg,augment):
    d=cfg['data'];e=cfg['experiment'];native=tuple(e['native_shape'])
    return MedicalVolumeDataset3D(root=d['root'],modality=e['modality'],target_shape=native,target_spacing_xyz=parse_float3(d.get('target_spacing')),
        patterns=d.get('patterns'),file_list=d.get('file_list'),augment_flip=augment,ct_min_hu=d.get('ct_min_hu',-1000),ct_max_hu=d.get('ct_max_hu',1500),ct_pad_hu=d.get('ct_pad_hu',-1000),
        mri_percentile_low=d.get('mri_percentile_low',1.0),mri_percentile_high=d.get('mri_percentile_high',99.0),mri_foreground_threshold=d.get('mri_foreground_threshold',0.0))

def _init_model(cfg,mode,init_checkpoint,device,startup=None):
    scope=startup.stage if startup is not None else lambda name:nullcontext()
    with scope('build AE and move to device'):
        model=build_autoencoder(cfg).to(device);info=None
    if mode in ('decoder','finetune'):
        ckpt=init_checkpoint or cfg.get('initialization',{}).get('checkpoint','')
        if not ckpt:raise ValueError(f'{mode} requires --init-checkpoint or initialization.checkpoint in config')
        with scope('load legacy AE weights'):
            info=initialize_from_checkpoint(model,ckpt,include_head=False,strict=False)
    set_training_mode(model,mode);return model,info


def train(config_path,run_dir,mode=None,init_checkpoint=None,resume=True):
    run=Path(run_dir);startup=StartupMonitor(run)
    rank=int(os.getenv('RANK','0'))
    try:
        with startup.stage('load training config'):
            cfg=load_yaml(config_path);tr=cfg['training'];loss_cfg=tr['losses'];mode=mode or tr.get('mode','scratch')
        with startup.stage('CUDA device and distributed initialization'):
            distributed,rank,local,world,device=setup_distributed(int(tr.get('seed',0)))
        with startup.stage('run directories and hardware metadata'):
            if is_main(rank):
                for folder in ('checkpoints','images','control'):(run/folder).mkdir(parents=True,exist_ok=True)
                atomic_json(run/'hardware.json',system_info());atomic_json(run/'control/status.json',{'status':'starting'})
        with startup.stage('initial rank barrier'):
            barrier()
        with startup.stage('dataset discovery'):
            dataset=make_dataset(cfg,True)
        with startup.stage('dataset split and file metadata'):
            ds,viz_ds,split_id=split_datasets(dataset,cfg,run if is_main(rank) else None)
        last=find_last(run) if resume else None
        # Resuming is self-contained; the old initialization file is not needed again.
        model_raw,init_info=_init_model(cfg,'scratch' if last else mode,init_checkpoint,device,startup)
        set_training_mode(model_raw,mode)
        with startup.stage('AE DDP setup / parameter broadcast'):
            model=wrap_ddp(model_raw,device,distributed)
        head=model_raw.head_name;use_dense=head=='clif'
        if is_main(rank) and not last:atomic_json(run/'initialization.json',{'mode':mode,'load':init_info})
        params=[p for p in model.parameters() if p.requires_grad]
        betas=tuple(float(x) for x in tr.get('adam_betas',([.5,.9] if use_dense else [.9,.999])))
        opt=(torch.optim.Adam if use_dense else torch.optim.AdamW)(params,lr=float(tr['lr']),betas=betas,weight_decay=float(tr.get('weight_decay',0)))
        amp=bool(tr.get('amp',True));amp_dtype=tr.get('amp_dtype','bf16');sc=scaler(device,amp,amp_dtype)
        # Block boundaries replace the old whole-AE replay. No parameter names
        # or backbone shapes change, so initialization weights remain usable.
        checkpoint_ae=bool(tr.get('activation_checkpointing',True))
        query_chunk=int(tr.get('training_query_chunk',16384))
        checkpoint_perceptual=bool(tr.get('checkpoint_perceptual',True))
        log_every=int(tr.get('log_every_steps',100));checkpoint_every=int(tr.get('checkpoint_every_steps',100))
        if log_every<1 or checkpoint_every<1:raise ValueError('log_every_steps and checkpoint_every_steps must be positive')
        if is_main(rank):log(run,f'[AE] block activation checkpointing={checkpoint_ae}; query_chunk={query_chunk}; perceptual_checkpointing={checkpoint_perceptual}')
        gan_w=float(loss_cfg.get('gan_weight',0));perc_w=float(loss_cfg.get('perceptual_weight',0));ssim_w=float(loss_cfg.get('ssim_weight',0))
        if not use_dense and (gan_w or perc_w or ssim_w):raise ValueError('Point-head training does not enable dense GAN/perceptual/SSIM losses; use CLIF or set these weights to zero')
        disc_raw=disc=disc_opt=disc_sc=perceptual=None
        if use_dense and gan_w>0:
            with startup.stage('discriminator construction and DDP setup'):
                dc=cfg.get('discriminator',{});disc_raw=PatchDiscriminator3D(1,int(dc.get('base_channels',32)),int(dc.get('layers',4)),int(dc.get('max_channels',256))).to(device);disc=wrap_ddp(disc_raw,device,distributed)
            disc_opt=torch.optim.Adam(disc.parameters(),lr=float(tr.get('discriminator_lr',tr['lr'])),betas=betas);disc_sc=scaler(device,amp,amp_dtype)
        if use_dense and perc_w>0:
            with startup.stage('perceptual network / pretrained weights'):
                pc=cfg.get('perceptual',{});perceptual=MONAIPerceptualLoss3D(pc.get('network_type','squeeze'),float(pc.get('fake_3d_ratio',.5)),pc.get('cache_dir') or None,bool(pc.get('pretrained',True))).to(device)
        epoch=1;cursor=0;step=0;best_psnr=-float('inf');best_ssim=-float('inf')
        sums={k:0. for k in ('l1','ssim','perceptual','kl','cov','gan_g','gan_d','adaptive','total')}
        count=0;epoch_seconds=0.
        if last:
            with startup.stage('restore training checkpoint and RNG'):
                c=torch.load(last,map_location='cpu',weights_only=False)
                if c.get('config_identity')!=digest(cfg) or c.get('split_identity')!=split_id or c.get('mode')!=mode:raise ValueError('Resume config/mode/dataset split changed; initialize a new run instead')
                model_raw.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);sc.load_state_dict(c['scaler'])
                if disc_raw is not None:
                    disc_raw.load_state_dict(c['discriminator'],strict=True);disc_opt.load_state_dict(c['discriminator_optimizer']);disc_sc.load_state_dict(c['discriminator_scaler'])
                epoch=c['epoch'];cursor=c['cursor'];step=c['global_step'];best_psnr=c['best_psnr'];best_ssim=c['best_ssim']
                restore_training_rng(c,rank,world,int(tr.get('seed',0)))
                progress=c.get('epoch_progress',[])
                if progress:
                    if len(progress)==world:
                        sums=progress[rank]['sums'];count=progress[rank]['count'];epoch_seconds=progress[rank]['seconds']
                    else:
                        # Keep the previous global epoch totals when GPU count changes.
                        sums={k:sum(p['sums'][k] for p in progress)/world for k in sums}
                        count=sum(p['count'] for p in progress)/world;epoch_seconds=max(p['seconds'] for p in progress)
                if is_main(rank):
                    trim_csv(run/'iterations.csv','global_step',step)
                    trim_csv(run/'metrics.csv','epoch',epoch-1)
                    log(run,f'[AE] RESUME {last}: epoch={epoch} cursor={cursor}/{len(ds)} global_step={step} best_psnr={best_psnr:.4f}')
                    if cursor and not progress:log(run,'[AE] Older checkpoint has no partial-epoch loss totals; this epoch reports only the remaining batches.')
        if is_main(rank):
            (run/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            if not last:log(run,f'[AE] START mode={mode} head={head} train_volumes={len(ds)} validation_volumes={len(viz_ds)} world_size={world}')
        native=tuple(cfg['experiment']['native_shape']);head_cfg=cfg['model']['head']['config']
        if is_main(rank):
            detail=f'random dense interior={head_cfg.get("training_patch",[64]*3)} halo={head_cfg.get("halo",6)}' if use_dense else f'scattered random points={tr.get("coords_per_batch",200000)} per volume; chunk={query_chunk}'
            log(run,f'[AE] supervision: {detail}; covariance_weight={loss_cfg.get("cov_weight",0)}')
        t0=time.time()
        def checkpoint(psnr_best=False):
            states=gather_rng()
            local_progress={'sums':dict(sums),'count':count,'seconds':epoch_seconds+time.time()-t0}
            progress=[local_progress]
            if distributed:
                progress=[None]*world;dist.all_gather_object(progress,local_progress)
            if not is_main(rank):return
            payload={'schema':3,'model':model_raw.state_dict(),'optimizer':opt.state_dict(),'scaler':sc.state_dict(),'epoch':epoch,'cursor':cursor,'global_step':step,'best_psnr':best_psnr,'best_ssim':best_ssim,'config':cfg,'config_identity':digest(cfg),'split_identity':split_id,'mode':mode,'world_size':world,'rng_states':states,'epoch_progress':progress,'batch_policy':'per_gpu'}
            if disc_raw is not None:payload.update(discriminator=disc_raw.state_dict(),discriminator_optimizer=disc_opt.state_dict(),discriminator_scaler=disc_sc.state_dict())
            save_ae(run,payload,psnr_best)
        stopped=False
        while epoch<=int(tr['epochs']):
            if coordinated_stop(run,device):stopped=True;break
            loader,batch_size=loader_for_epoch(ds,tr,rank,world,epoch,cursor)
            model.train()
            if mode=="decoder":model_raw.encoder.eval()
            if disc is not None:disc.train()
            if perceptual is not None:perceptual.eval()
            t0=time.time();segment_count=0
            total_batches=math.ceil(len(ds)/(batch_size*world))
            if is_main(rank):log(run,f'[AE] epoch={epoch}/{tr["epochs"]} start batch={math.ceil(cursor/(batch_size*world))}/{total_batches} global_step={step}')
            for batch in startup.batches(loader):
                if coordinated_stop(run,device):stopped=True;break
                x=batch['volume'].to(device,non_blocking=True);b=x.shape[0];opt.zero_grad(set_to_none=True)
                if use_dense:
                    patch=head_cfg.get('training_patch',[48,48,48]);halo=int(head_cfg.get('halo',8));coords,cells,_=random_dense_patch_coordinates(b,native,patch,halo,float(head_cfg.get('scale_min',1.0)),float(head_cfg.get('scale_max',1.0)),device,x.dtype)
                    with autocast(device,amp,amp_dtype):
                        pred_h,mean,raw=model(x,coords,cells,sample_latent=True,training_outputs=True,activation_checkpointing=checkpoint_ae,training_query_chunk=query_chunk)
                        post=DiagonalGaussian3D(mean,raw)
                        target_h=sample_dense_patch(x,coords);pred=crop_halo(pred_h,halo);target=crop_halo(target_h,halo);l1=l1_loss(pred,target);kl=post.kl()
                    if segment_count==0:
                        print(f'[AE rank={rank}] CLIF shapes: input={tuple(x.shape)}; rendered_with_halo={tuple(pred_h.shape)}; loss_patch={tuple(pred.shape)}; KL=full latent',flush=True)
                    fp32=torch.autocast('cuda',enabled=False) if device.type=='cuda' else nullcontext()
                    with fp32:
                        cov=stable_covariance_loss(post.mode().float(),float(loss_cfg.get('cov_eps',1e-3)),float(loss_cfg.get('cov_value_clip',10.0))) if float(loss_cfg.get('cov_weight',0)) else pred.new_zeros(())
                        perc=loss_checkpoint(perceptual,pred.float(),target.float(),enabled=checkpoint_perceptual) if perceptual is not None else pred.new_zeros(())
                        ssim=ssim3d_loss(pred.float(),target.float()) if ssim_w>0 else pred.new_zeros(())
                    gan_active=disc is not None and step>=int(loss_cfg.get('gan_start_step',0));gan_d=pred.new_zeros(());gan_g=pred.new_zeros(());adapt=pred.new_zeros(())
                    if gan_active:
                        set_requires_grad(disc,True);disc_opt.zero_grad(set_to_none=True)
                        with autocast(device,amp,amp_dtype):gan_d=hinge_discriminator_loss(disc(target.detach()),disc(pred.detach()))
                        disc_sc.scale(gan_d).backward();disc_sc.unscale_(disc_opt)
                        if tr.get('discriminator_grad_clip',1)>0:torch.nn.utils.clip_grad_norm_(disc.parameters(),float(tr['discriminator_grad_clip']))
                        disc_sc.step(disc_opt);disc_sc.update();set_requires_grad(disc,False)
                        with autocast(device,amp,amp_dtype):gan_g=hinge_generator_loss(disc_raw(pred))
                    rec=float(loss_cfg.get('l1_weight',1))*l1+perc_w*perc+ssim_w*ssim
                    if gan_active:
                        adapt=adaptive_gan_weight(rec,gan_g,model_raw.get_last_layer_weight(),float(loss_cfg.get('adaptive_gan_eps',1e-4)),float(loss_cfg.get('adaptive_gan_max',1e4))) if bool(loss_cfg.get('adaptive_gan_weight',True)) else pred.new_tensor(1.0)
                    total=rec+float(loss_cfg.get('kl_weight',1e-4))*kl+float(loss_cfg.get('cov_weight',0))*cov+(gan_w*adapt*gan_g if gan_active else 0)
                else:
                    train_res=tr.get('train_resolutions',[list(native)]);effective=choose_resolution(native,train_res);n=int(tr.get('coords_per_batch',200000));coords=sample_liif_coordinates(b,n,device,x.dtype,float(tr.get('coord_jitter',0)));cells=cells_for_resolution(effective,b,n,device,x.dtype) if bool(head_cfg.get('use_cell',True)) else None
                    align=False;target=sample_volume_points(x,coords,align_corners=align)
                    with autocast(device,amp,amp_dtype):
                        pred,mean,raw=model(x,coords,cells,sample_latent=True,training_outputs=True,activation_checkpointing=checkpoint_ae,training_query_chunk=query_chunk)
                        post=DiagonalGaussian3D(mean,raw);l1=l1_loss(pred,target);kl=post.kl()
                    fp32=torch.autocast('cuda',enabled=False) if device.type=='cuda' else nullcontext()
                    with fp32:cov=stable_covariance_loss(post.mode().float(),float(loss_cfg.get('cov_eps',1e-3)),float(loss_cfg.get('cov_value_clip',10.0))) if float(loss_cfg.get('cov_weight',0)) else pred.new_zeros(())
                    ssim=perc=gan_g=gan_d=adapt=pred.new_zeros(());total=float(loss_cfg.get('l1_weight',1))*l1+float(loss_cfg.get('kl_weight',1e-4))*kl+float(loss_cfg.get('cov_weight',0))*cov
                if not torch.isfinite(total):raise FloatingPointError(f'non-finite AE loss at step {step}')
                sc.scale(total).backward();sc.unscale_(opt)
                if tr.get('grad_clip',1)>0:torch.nn.utils.clip_grad_norm_(params,float(tr['grad_clip']))
                sc.step(opt);sc.update();
                if disc is not None:set_requires_grad(disc,True)
                step+=1;count+=1;segment_count+=1;cursor=min(len(ds),cursor+batch_size*world)
                vals={'l1':l1,'ssim':ssim,'perceptual':perc,'kl':kl,'cov':cov,'gan_g':gan_g,'gan_d':gan_d,'adaptive':adapt,'total':total}
                for k,value in vals.items():sums[k]+=float(value.detach())
                if step==1 or step%log_every==0 or cursor==len(ds):
                    av=reduce_metrics(sums,count,device)
                    if is_main(rank):
                        done=math.ceil(cursor/(batch_size*world));seconds_per_step=(time.time()-t0)/segment_count
                        row={'epoch':epoch,'global_step':step,'batch':done,'batches':total_batches,'progress_pct':100*cursor/len(ds),'epoch_seconds':epoch_seconds+time.time()-t0,'eta_seconds':max(0,total_batches-done)*seconds_per_step,**av,
                             'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.}
                        append_csv(run/'iterations.csv',row)
                        log(run,f'[AE] epoch={epoch}/{tr["epochs"]} batch={done}/{total_batches} ({row["progress_pct"]:.1f}%) step={step} loss={av["total"]:.6g} l1={av["l1"]:.6g} kl={av["kl"]:.6g} gan={av["gan_g"]:.6g} perceptual={av["perceptual"]:.6g} peak_gpu={row["peak_gpu_gib"]:.2f}GiB ETA={row["eta_seconds"]:.0f}s')
                        atomic_json(run/'control/status.json',{'status':'running',**row,'world_size':world})
                # Free training tensors and gradients before monitoring; do not
                # keep the final volume/graph alive alongside a preview volume.
                opt.zero_grad(set_to_none=True)
                if disc_opt is not None:disc_opt.zero_grad(set_to_none=True)
                del x,pred,post,mean,raw,coords,cells,target,vals,total,l1,kl,ssim,perc,cov,gan_g,gan_d,adapt
                if use_dense:del pred_h,target_h,rec
                if step%checkpoint_every==0:checkpoint()
            checkpoint()
            if stopped:break
            av=reduce_metrics(sums,count,device);monitor=[None]
            if is_main(rank):
                try:
                    with autocast(device,amp,amp_dtype):monitor[0]={'metrics':validate(model_raw,viz_ds,cfg,device,run)}
                except Exception:monitor[0]={'error':traceback.format_exc()}
            if distributed:dist.broadcast_object_list(monitor,src=0)
            if 'error' in monitor[0]:raise RuntimeError(monitor[0]['error'])
            vm=monitor[0]['metrics']
            if vm is None:stopped=True;break
            completed_epoch=epoch
            # The epoch is committed only after its preview completes. A stop
            # during monitoring resumes here, with no training batches replayed.
            preview_result=[None]
            if is_main(rank):
                every=int(cfg['validation'].get('preview_every_epochs',1))
                if every>0 and completed_epoch%every==0:
                    if seconds_left()<=float(cfg['validation'].get('preview_reserve_seconds',600)):
                        preview_result[0]={'paused':True}
                        log(run,f'[AE] epoch={epoch}: preview deferred until resume (job time limit).')
                    else:
                        rng=capture_rng()
                        try:
                            index=random.Random(int(cfg['validation'].get('seed',91415))+completed_epoch).randrange(len(viz_ds))
                            sample=viz_ds[index];source=sample.get('path',str(index));vx=sample['volume'].unsqueeze(0).to(device)
                            image_path=run/'images'/f'epoch_{completed_epoch:06d}.png'
                            log(run,f'[AE] epoch={epoch}: preview volume={source}')
                            with autocast(device,amp,amp_dtype):
                                saved=save_ae_preview(model_raw,vx,native,image_path,int(head_cfg.get('render_chunk',65536)),tuple(head_cfg.get('render_tile',[32,64,64])),int(head_cfg.get('halo',8)),lambda:should_stop(run),title=f'{head} | epoch {epoch} | {Path(source).name}')
                            if saved:
                                atomic_json(image_path.with_suffix('.json'),{'epoch':epoch,'global_step':step,'validation_index':index,'source':source,'latent':'posterior_mean','planes':'central axial, coronal, sagittal'})
                                log(run,f'[AE] preview saved: {image_path}')
                            else:preview_result[0]={'paused':True}
                            del vx,sample
                        except Exception:preview_result[0]={'error':traceback.format_exc()}
                        finally:restore_rng(rng)
            if distributed:dist.broadcast_object_list(preview_result,src=0)
            if preview_result[0]:
                if 'error' in preview_result[0]:raise RuntimeError(preview_result[0]['error'])
                stopped=True;break
            better_psnr=vm['psnr']>best_psnr
            best_psnr=max(best_psnr,vm['psnr']);best_ssim=max(best_ssim,vm['ssim'])
            if is_main(rank):
                row={'epoch':completed_epoch,'global_step':step,'seconds':epoch_seconds+time.time()-t0,'world_size':world,'global_batch':batch_size*world,**av,**{'val_patch_'+k:v for k,v in vm.items()}}
                if device.type=='cuda':row['peak_gpu_gib']=torch.cuda.max_memory_allocated(device)/2**30
                append_csv(run/'metrics.csv',row)
                log(run,f'[AE] epoch={completed_epoch}/{tr["epochs"]} complete step={step} loss={av["total"]:.6g} val_patch_psnr={vm["psnr"]:.4f} val_patch_ssim={vm["ssim"]:.5f} seconds={row["seconds"]:.1f} best_psnr={best_psnr:.4f}')
            epoch+=1;cursor=0;sums={k:0. for k in sums};count=0;epoch_seconds=0.;t0=time.time()
            checkpoint(better_psnr)
            if is_main(rank):
                log(run,'[AE] saved checkpoints/last.pt'+(' and checkpoints/best.pt (validation PSNR improved)' if better_psnr else ''))
                atomic_json(run/'control/status.json',{'status':'running','epoch':completed_epoch,'global_step':step,'world_size':world})
            barrier()
        checkpoint()
        if is_main(rank):
            atomic_json(run/'control/status.json',{'status':'paused' if stopped else 'complete','epoch':epoch,'cursor':cursor,'global_step':step,'world_size':world})
            log(run,f'[AE] {"PAUSED" if stopped else "COMPLETE"} next_epoch={epoch} cursor={cursor} global_step={step}; last.pt is ready to resume')
    except Exception:
        if is_main(rank):
            log(run,'[AE] FAILED\n'+traceback.format_exc())
            atomic_json(run/'control/status.json',{'status':'failed','error':traceback.format_exc()})
        raise
    finally:
        with startup.stage('distributed cleanup'):
            cleanup()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--mode',choices=['scratch','decoder','finetune']);p.add_argument('--init-checkpoint');p.add_argument('--no-resume',action='store_true');a=p.parse_args();train(a.config,a.run_dir,a.mode,a.init_checkpoint,not a.no_resume)
if __name__=='__main__':main()
