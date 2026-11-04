import argparse,copy,json,math,os,time,random,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from .common import *
from .data import Scenes,core_mask,to_device
from .network import Network
from .pg import Decoder
from .metrics import arrays,aggregate
from space_v2.resources import wait_headroom

MILESTONES=[0,5000,10000,20000,40000,60000,80000]
def lr_at(step,cfg):
    if step<cfg['warmup']:return cfg['lr']*(step+1)/cfg['warmup']
    return cfg['lr']*(cfg['min_lr_ratio']+(1-cfg['min_lr_ratio'])*.5*(1+math.cos(math.pi*min(1,(step-cfg['warmup'])/(cfg['horizon']-cfg['warmup'])))))

def q_prediction(model,decoder,case,mask,amp=True):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        pred=model(case['y'][None],case['b'][None],case['valid'][None])[0]
    return decoder(pred[mask],case['b'].expand_as(case['y'])[mask]),pred[mask]

def objective(model,decoder,case,ordinal):
    mask=core_mask(case);positions=mask.nonzero()
    if not len(positions):return None,None
    if len(positions)>2048:
        gen=torch.Generator(device='cpu').manual_seed(500_000_000+ordinal)
        positions=positions[torch.randperm(len(positions),generator=gen)[:2048].to(positions.device)]
        mask=torch.zeros_like(mask);mask[tuple(positions.T)]=True
    lq,_=q_prediction(model,decoder,case,mask)
    with torch.no_grad():lp=decoder(case['s'][mask],case['b'].expand_as(case['y'])[mask])
    value=-(lp.exp()*lq).sum(-1).mean()
    return value,dict(candidates=len(positions),KL=float((lp.exp()*(lp-lq)).sum(-1).mean().detach()),scene=case['scene'])

@torch.no_grad()
def evaluate(model,decoder,data,limit=None,amp=True):
    model.eval();rows=[];cal=[];inputs=[]
    for realization in range(2):
        for i in range(len(data.rows) if limit is None else min(limit,len(data.rows))):
            for context in [128,256]:
                wait_headroom(1)
                case=to_device(data.sample(i+realization*len(data.rows),context,force_natural=True));mask=core_mask(case)
                n=int(mask.sum())
                if not n:continue
                q,s=q_prediction(model,decoder,case,mask,amp=amp)
                b=case['b'].expand_as(case['y'])[mask]
                p=decoder(case['s'][mask],b)
                pieces=[]
                for begin in range(0,n,2048):
                    sl=slice(begin,begin+2048)
                    a,c,bound=arrays(q[sl].cpu().numpy(),p[sl].cpu().numpy(),case['v'][mask][sl].cpu().numpy(),
                                     b[sl].cpu().numpy(),signal=s[sl].cpu().numpy())
                    pieces.append(a);cal.extend(c)
                merged={k:sum(x[k]*x['n'] for x in pieces)/n for k in pieces[0] if k!='n'}
                digest=hashlib.sha256()
                for name in ['y','v','b','valid']:digest.update(case[name].cpu().numpy().tobytes())
                rows.append(dict(scene=case['scene'],realization=realization,context=context,n=n,input_sha256=digest.hexdigest(),**merged))
    model.train()
    return dict(summary=aggregate(rows),rows=rows,calibration=cal,precision='BF16 backbone/FP32 PG' if amp else 'FP32 TF32off',
                nominal_mothers=len(data.rows),nonempty_mothers=len({r['scene'] for r in rows}))

def save(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)

def state(model,ema,opt,cfg,step,cursor,best,counts,seen,elapsed):
    return dict(config=cfg,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,
        data_cursor=cursor,best=best,counts=counts,seen=sorted(seen),elapsed_seconds=elapsed,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        lr_reconstruction='lr_at(current_step,config) for next update; no hidden scheduler state')

def seed_all():
    torch.set_num_threads(4);torch.manual_seed(7103);np.random.seed(7103);random.seed(7103)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True

def one_update(model,ema,opt,decoder,train,step,cursor,cfg,counts,seen):
    wait_headroom(1);model.train();opt.zero_grad(set_to_none=True);lr=lr_at(step,cfg)
    for g in opt.param_groups:g['lr']=lr
    records=[]
    while not records:
        for micro in range(cfg['accumulation']):
            ordinal=cursor;cursor+=1;case=to_device(train.sample(ordinal,cfg['contexts'][step%2]))
            value,info=objective(model,decoder,case,ordinal)
            counts['natural' if case['domain']=='natural' else 'auxiliary']+=1;counts['core_positions']+=4*64**2
            seen.add(case['scene'])
            if value is None:counts['empty_cores']+=1;continue
            assert bool(torch.isfinite(value)),f'Nonfinite loss at update{step}'
            (value/cfg['accumulation']).backward();records.append(info);counts['candidates']+=info['candidates']
        if not records:counts['empty_batches']+=1
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True);opt.step()
    with torch.no_grad():
        for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,1-cfg['ema'])
    return cursor,records,float(norm)

def verify_sources(cfg):
    assert sha(ROOT/'configs/ptc_s5m2_iso12800_localRG_locked.json') == cfg['ptc_sha256']
    assert sha(ROOT/'datasets/source_manifest.json') == cfg['dataset_manifest_sha256']
    assert sha(DATA/'MANIFEST.json') == cfg['manifest_sha256']
    frozen=json.loads((ROOT/'datasets/source_manifest.json').read_text(encoding='utf-8-sig'))['rows']
    actual=json.loads((DATA/'MANIFEST.json').read_text(encoding='utf-8-sig'))['rows']
    keys=('path','sha256','split','scene_group','cache_key')
    assert len(frozen)==len(actual) and all(tuple(x.get(k) for k in keys)==tuple(y.get(k) for k in keys) for x,y in zip(frozen,actual))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/train.json').read_text())
    cfg.update(manifest_sha256=sha(DATA/'MANIFEST.json'))
    verify_sources(cfg)
    lock=OUT/'TRAIN.lock';proc=psutil.Process()
    if lock.exists():
        old=json.loads(lock.read_text())
        try:
            live=psutil.Process(old['pid'])
            if abs(live.create_time()-old['created'])<.01:raise RuntimeError('Healthy ISO12800 worker already active')
        except psutil.NoSuchProcess:pass
    write_json(lock,dict(pid=os.getpid(),created=proc.create_time(),time=now()))
    try:
        seed_all()
        if not (OUT/'PG_LUT.npz').exists():
            from .pg import build
            build()
        decoder=Decoder();model=Network('N0','PG').cuda();ema=copy.deepcopy(model).eval()
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        train=Scenes('train',7103);dev=Scenes('dev',7103);step=cursor=0;best=float('inf');elapsed=0.
        counts=dict(natural=0,auxiliary=0,candidates=0,empty_cores=0,core_positions=0,empty_batches=0);seen=set()
        latest=OUT/'LATEST_FULL.pt'
        if latest.exists():
            st=torch.load(latest,map_location='cpu',weights_only=False);assert st['config']==cfg
            model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer'])
            step=st['step'];cursor=st['data_cursor'];best=st['best'];counts=st['counts'];seen=set(st['seen']);elapsed=st['elapsed_seconds']
            torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng']);np.random.set_state(st['numpy_rng']);random.setstate(st['python_rng']);del st
        else:
            s=state(model,ema,opt,cfg,0,0,best,counts,seen,0.);save(OUT/'CHECKPOINT_00000_FULL.pt',s);save(latest,s)
            baseline=evaluate(ema,decoder,dev,limit=4);write_json(OUT/'INITIAL_DEV_SMOKE.json',baseline)
        started=time.perf_counter();origin=step
        while step<80000:
            cursor,records,norm=one_update(model,ema,opt,decoder,train,step,cursor,cfg,counts,seen);step+=1
            if step%20==0 or step==1:
                progress=dict(status='TRAINING',step=step,target=80000,lr=lr_at(step-1,cfg),counts=counts,data_cursor=cursor,
                    recent_KL=float(np.mean([r['KL'] for r in records])),grad_norm=norm,time=now(),pid=os.getpid(),created=proc.create_time(),
                    elapsed_seconds=elapsed+time.perf_counter()-started,seconds_per_update=(time.perf_counter()-started)/(step-origin),
                    source='local RG dark calibrated RN, JOT gain/FWC; nominal512',test_accessed=False)
                write_json(OUT/'TRAIN_7103_PROGRESS.json',progress)
                if step%200==0 or step==1:print(json.dumps(progress),flush=True)
            do_eval=step%2000==0 or step in MILESTONES
            if do_eval:
                r=evaluate(ema,decoder,dev);criterion=r['summary']['mother_equal']['KL']
                r.update(step=step,selection_metric='mother-equal Oracle KL, fixed K304 + genuine tail')
                write_json(OUT/'METRICS_DEV.json',r)
                with (OUT/'DEV_HISTORY.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(step=step,**r['summary']['mother_equal']))+'\n')
                if criterion<best:
                    best=criterion;save(OUT/'BEST_WEIGHTS.pt',dict(model=ema.state_dict(),config=cfg,step=step,criterion=best));write_json(OUT/'BEST_METRICS_DEV.json',r)
                if step in MILESTONES:write_json(OUT/f'DEV_{step:05d}.json',r)
            if step%500==0 or do_eval:
                s=state(model,ema,opt,cfg,step,cursor,best,counts,seen,elapsed+time.perf_counter()-started);save(latest,s)
                if step in MILESTONES:save(OUT/f'CHECKPOINT_{step:05d}_FULL.pt',s)
        verify_sources(cfg)
        write_json(OUT/'TRAINING_COMPLETE.json',dict(status='80K_TRAINING_COMPLETE',time=now(),step=80000,best_KL=best,
                   best_sha256=sha(OUT/'BEST_WEIGHTS.pt'),latest_full_sha256=sha(latest),single_seed=True))
    except BaseException as exc:
        import traceback
        write_json(OUT/'FAILURE.json',dict(status='FAILED',time=now(),error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('pid')==os.getpid():lock.unlink()

if __name__=='__main__':run()
