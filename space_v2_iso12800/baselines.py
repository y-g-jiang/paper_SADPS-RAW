import json,math,time,hashlib
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.ndimage import uniform_filter
from baselines.local_pg import lambda_grid,prior_weights
from space_v2.context_baselines import histogram_at
from space_v2.cpu_pg import adaptive
from .common import OUT,G,RN,K,UPPER,now,write_json,sha,ROOT
from .data import Scenes,core_mask
from .pg import Decoder
from .metrics import arrays,aggregate

def decoder_numpy(dec,s,b):
    s,b=np.broadcast_arrays(np.asarray(s,np.float32),np.asarray(b,np.float32))
    out=[]
    for i in range(0,s.size,2048):
        with torch.no_grad():out.append(dec(torch.from_numpy(s.ravel()[i:i+2048]),torch.from_numpy(b.ravel()[i:i+2048])).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,K+2))

def fit_train():
    p=OUT/'TRAIN_GLOBAL_AND_PRIOR.json'
    if p.exists():return json.loads(p.read_text())
    data=Scenes('train',7103);dec=Decoder('cpu');acc=np.full((4,K+2),-np.inf);counts=np.zeros(4,np.int64);sums=np.zeros(4)
    n=zero=0;su=ss=0.
    for i in range(len(data.rows)):
        case=data.sample(i,128,force_natural=True);mask=core_mask(case).numpy();sig=case['s'].numpy();b=case['b'].numpy()
        h,w=sig.shape[-2:];core=(slice(None),slice((h-64)//2,(h+64)//2),slice((w-64)//2,(w+64)//2))
        valid=np.broadcast_to(case['valid'].numpy(),sig.shape)[core]>0;v=sig[core][valid]/G
        zero+=int((v==0).sum());v=v[v>0];n+=len(v);su+=float(v.sum(dtype=np.float64));ss+=float(np.square(v.astype(float)).sum())
        for c in range(4):
            if not mask[c].any():continue
            s=sig[c][mask[c]];q=decoder_numpy(dec,s,float(b[c,0,0]))
            acc[c]=np.logaddexp(acc[c],logsumexp(q,axis=0));counts[c]+=len(s);sums[c]+=float(s.sum(dtype=np.float64))
    mean=su/n;variance=max(ss/n-mean**2,1e-8)
    result=dict(time=now(),train_mothers=len(data.rows),source='same ISO12800 generator, train only',
        per_CFA_logq=(acc-np.log(counts[:,None])).tolist(),candidate_counts=counts.tolist(),signal_DN=(sums/counts).tolist(),
        prior=dict(zero_mass=zero/(n+zero),gamma_shape=mean**2/variance,gamma_scale_e=variance/mean),
        window_width=3,prior_fit='all valid train central signals (not zero-conditioned), gamma moment fit, exact point-mass zero')
    write_json(p,result);return result

def local_posterior(y,valid,mask,b,prior):
    yy=np.asarray(y,np.int64);valid=np.asarray(valid,bool);mask=np.asarray(mask,bool);centers=np.argwhere(mask)
    width=3;means=uniform_filter(yy.astype(float)*valid,width,mode='constant')/np.maximum(uniform_filter(valid.astype(float),width,mode='constant'),1e-12)


    end=max(128.,1.8*float(means[mask].max())+6*RN)
    grid=np.unique(np.r_[np.arange(0,32,.1),np.arange(32,end,.5),end])/G
    codes=np.unique(yy[valid]);ll=np.empty((len(grid),len(codes)))
    for j,k in enumerate(codes):
        event='cdf' if k==0 else 'upper' if k==UPPER else 'bin'
        threshold=.5 if k==0 else UPPER-.5 if k==UPPER else k
        ll[:,j]=adaptive(grid,threshold,b,G,RN,event)[0]
    hist=histogram_at(yy,valid,centers,width,codes)
    lw=hist@ll.T+np.log(prior_weights(grid,prior));lw-=logsumexp(lw,axis=1)[:,None]
    assert np.exp(lw[:,-2:]).sum(1).max()<1e-5,'Local lambda grid boundary mass: diagnose before claiming complete baseline'
    dec=Decoder('cpu');kernels=decoder_numpy(dec,grid*G,b)
    out=np.empty((len(centers),K+2))
    for k in range(K+2):out[:,k]=logsumexp(lw+kernels[:,k][None],axis=1)
    out-=logsumexp(out,axis=1)[:,None]
    return out

def foi_signal(y,valid,b):


    from types import ModuleType
    path=ROOT/'space_v2/foi_natural.py';text=path.read_text(encoding='utf-8')
    text=text.replace('from .common import G,RN','from space_v2_iso12800.common import G,RN')

    text=text.replace('np.clip(code,0,16383)',f'np.clip(code,0,{UPPER})')
    text=text.replace('transform=FoiTransform(G,RN,b,maximum=maximum)',
        "maximum=max(maximum,G*(poisson.ppf(1-1e-13,max(32.,float(observed.max())+16)/G)+2)+12*RN+4+max(b,0))\n    transform=FoiTransform(G,RN,b,maximum=maximum)")
    scope={'__name__':'space_v2_iso12800._foi','__package__':'space_v2_iso12800'}
    exec(compile(text,str(path),'exec'),scope)
    return scope['infer'](y,valid,b)[0]

def evaluate_baselines(limit=None,include_foi=True,job_rank=0,job_count=1):
    fit=fit_train();data=Scenes('dev',7103);dec=Decoder('cpu');rows=[];folder=OUT/'BASELINE_CASES';folder.mkdir(exist_ok=True)
    for realization in range(2):
        for i in range(len(data.rows) if limit is None else min(limit,len(data.rows))):
            for context in [128,256]:
                job=(realization*len(data.rows)+i)*2+[128,256].index(context)
                if job%job_count!=job_rank:continue
                key=f'{i:03d}_r{realization}_c{context}';dest=folder/(key+('_all' if include_foi else '_early')+'.json')
                if dest.exists():rows.extend(json.loads(dest.read_text())['rows']);continue
                case=data.sample(i+realization*len(data.rows),context,force_natural=True);mask=core_mask(case).numpy()
                digest=hashlib.sha256()
                for name in ['y','v','b','valid']:digest.update(case[name].numpy().tobytes())
                records=[]
                for c in range(4):
                    m=mask[c];n=int(m.sum())
                    if not n:continue
                    b=float(case['b'][c,0,0]);signal=case['s'][c].numpy()[m];v=case['v'][c].numpy()[m]
                    teacher=decoder_numpy(dec,signal,b);y=case['y'][c].numpy();valid=case['valid'][0].numpy()>0
                    preds={'G-PMF':np.broadcast_to(np.array(fit['per_CFA_logq'][c]),(n,K+2)),
                           'LB3':local_posterior(y,valid,m,b,fit['prior'])}
                    if include_foi:preds['F-qPG']=decoder_numpy(dec,foi_signal(y,valid,b)[m],b)
                    for method,q in preds.items():
                        pieces=[]
                        for a in range(0,n,2048):
                            sl=slice(a,a+2048);met,cal,bound=arrays(q[sl],teacher[sl],v[sl],b)
                            pieces.append(met)
                        record=dict(method=method,scene=case['scene'],context=context,realization=realization,CFA=c,n=n,
                            **{k:sum(x[k]*x['n'] for x in pieces)/n for k in pieces[0] if k!='n'})
                        records.append(record)

                combined=[]
                for method in sorted({r['method'] for r in records}):
                    rr=[r for r in records if r['method']==method];n=sum(r['n'] for r in rr)
                    combined.append(dict(method=method,scene=case['scene'],context=context,realization=realization,n=n,input_sha256=digest.hexdigest(),
                        **{k:sum(r[k]*r['n'] for r in rr)/n for k in rr[0] if k not in ['method','scene','context','realization','CFA','n']}))
                write_json(dest,dict(rows=combined));rows.extend(combined)
                write_json(OUT/'BASELINE_PROGRESS.json',dict(time=now(),case=key,full_dev=limit is None,include_foi=include_foi))
    return dict(rows=rows,summaries={m:aggregate([r for r in rows if r['method']==m]) for m in sorted({r['method'] for r in rows})})

def early_gate(model,decoder,train,dev,evaluation):
    fit=fit_train();r=evaluate_baselines(limit=4,include_foi=False)
    initial=json.loads((OUT/'INITIAL_DEV_SMOKE.json').read_text())['summary']['mother_equal']['KL']
    from .train import evaluate
    matched=evaluate(model,decoder,dev,limit=4)
    current=matched['summary']['mother_equal']['KL']
    assert np.isfinite(current) and current<initial,'20k matched diagnostic not decreasing: investigate without changing schedule'
    result=dict(status='CONTINUE_ORIGINAL_80K_SCHEDULE',time=now(),step=20000,
        comparison='first4 dev mothers x2 noises x128/256, fixed early diagnostic; complete548 N0 dev also retained',
        initial_N0_KL=initial,current_N0=matched['summary'],early_baselines=r['summaries'],
        full_dev=evaluation['summary'],train_fit_sha256=sha(OUT/'TRAIN_GLOBAL_AND_PRIOR.json'),
        oracle_reference='teacher is exact known-lambda conditional PG; oracle-vs-oracle KL=0, coarsenedK304',
        schedule_changed=False,needs_to_win_baselines=False)
    write_json(OUT/'GATE_20000.json',result);return result
