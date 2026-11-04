import inspect
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.ndimage import binary_dilation
from .common import G, RN, K
from . import remote_gpu_math as original
from .metrics import arrays
from space_v2.cpu_pg import adaptive


def _adapt():
    source=inspect.getsource(original.local_laws)
    source=source.replace('def local_laws(y,valid,mask,b,prior,decoder):',
        'def flexible_local(y,valid,mask,b,prior,decoder,width=3,upper=UPPER):')
    source=source.replace('valid,3,mode=', 'valid,width,mode=').replace('float),3,mode=', 'float),width,mode=')
    source=source.replace('histogram_at(yy,valid,centers,3,codes)', 'histogram_at(yy,valid,centers,width,codes)')
    source=source.replace('codes!=UPPER','codes!=upper').replace('(UPPER,UPPER-.5,','(upper,upper-.5,')
    source=source.replace('codes=np.unique(yy[valid]);',
        'support=binary_dilation(mask,structure=np.ones((width,width),bool))&valid; codes=np.unique(yy[support]);')
    source=source.replace("return {'LB3':lb,'L3':local},dict(",
        "return {'LB':lb,'Local':local},dict(component_grid_DN=grid*G,component_logweights=lw,component_logkernels=kernels,MLE_signal_DN=lam*G,")
    assert 'uniform_filter(yy.astype(float)*valid,width,' in source
    assert 'uniform_filter(valid.astype(float),width,' in source
    scope=dict(vars(original));scope['binary_dilation']=binary_dilation
    exec(compile(source,__file__+':explicit_window_adapter','exec'),scope)
    return scope['flexible_local']


flexible_local=_adapt()


def score(q,p,v,b,signal=None,components=None):





    q=np.asarray(q,np.float64);q=q-logsumexp(q,axis=1)[:,None]
    p=np.asarray(p,np.float64);p=p-logsumexp(p,axis=1)[:,None]
    v=np.asarray(v);deep=v < -K
    safe=np.maximum(v,-K)
    values,cal,bound=arrays(q,p,safe,b)
    Q=np.exp(q);P=np.exp(p);w=-v.astype(int);codes=-np.arange(K+2,dtype=float)
    nll=-q[np.arange(len(v)),np.minimum(w,K+1)]
    qc=Q.cumsum(1)
    crps=((qc[:,:K+1]-(np.arange(K+1)[None]>=w[:,None]))**2).sum(1)+np.maximum(w-K-1,0)
    if deep.any():
        biases=np.broadcast_to(np.asarray(b,float),v.shape)
        if components is not None:
            grid,lw,kernels=components
            for i in np.flatnonzero(deep):
                den=adaptive(np.asarray(grid)/G,-K-.5,float(biases[i]),G,RN,'cdf')[0]
                mass=adaptive(np.asarray(grid)/G,int(v[i]),float(biases[i]),G,RN,'bin')[0]
                norm=logsumexp(lw[i]+logsumexp(np.asarray(kernels,float),axis=1))
                nll[i]=-(logsumexp(lw[i]+kernels[:,-1]+mass-den)-norm)
        else:
            assert signal is not None,'A deep integer code requires an explicit tail law'
            ss=np.broadcast_to(np.asarray(signal,float),v.shape)
            for i in np.flatnonzero(deep):
                den=float(adaptive(ss[i]/G,-K-.5,float(biases[i]),G,RN,'cdf')[0])
                mass=float(adaptive(ss[i]/G,int(v[i]),float(biases[i]),G,RN,'bin')[0])
                nll[i]=-(q[i,-1]+mass-den)


    mu=Q@codes;omu=P@codes
    lo=(qc<.05).sum(1);hi=(qc<.95).sum(1)
    values.update(NLL=float(nll.mean()),CRPS=float(crps.mean()),mean_MSE=float(np.mean((mu-v)**2)),
        oracle_mean_MSE=float(np.mean((omu-v)**2)),coverage90=float(np.mean((w>=lo)&(w<=hi))))

    for row in cal:
        thresholds={'zero':0,'negative':1,'le_minus2':2,'le_minus3':3,'le_minus4':4,'le_minus6':6,'le_minus8':8}
        t=thresholds[row['event']];prob=Q[:,0] if t==0 else Q[:,t:].sum(1)
        keep=np.minimum((prob*10).astype(int),9)==row['bin']
        row['observed']=int(((w==0) if t==0 else (w>=t))[keep].sum())
    return values,cal,bound


def admission(case,decoder,prior_fit=None):

    from .common import OUT
    from .data import core_mask
    from .baselines import local_posterior,decoder_numpy
    from .remote_gpu_math import local_laws
    from space_v2.integer_scoring import IntegerLaw
    import json
    fit=prior_fit if prior_fit is not None else json.loads((OUT/'TRAIN_GLOBAL_AND_PRIOR.json').read_text())
    mask=core_mask(case).numpy();c=next(c for c in range(4) if mask[c].any())

    m=np.zeros_like(mask[c]);coords=np.argwhere(mask[c])[:8];m[tuple(coords.T)]=True
    y=case['y'][c].numpy();valid=case['valid'][0].numpy()>0;b=float(case['b'][c,0,0])
    q,meta=flexible_local(y,valid,m,b,fit['prior'],decoder,3)
    base,_=local_laws(y,valid,m,b,fit['prior'],decoder)
    err=float(abs(q['LB']-base['LB3']).max());assert err<1e-8
    fixed=local_posterior(y,valid,m,b,fit['prior']);err2=float(abs(q['LB']-fixed).max());assert err2<1e-8
    s=case['s'][c].numpy()[m];p=decoder_numpy(decoder,s,b);v=case['v'][c].numpy()[m]
    a=score(q['LB'],p,v,b)[0];ref=arrays(q['LB'],p,v,b)[0]
    assert max(abs(a[k]-ref[k]) for k in ['KL','J_CDF','NLL','CRPS','mean_MSE'])<1e-10


    label=-400
    comp=(meta['component_grid_DN'],meta['component_logweights'][:1],meta['component_logkernels'])
    got=score(q['LB'][:1],q['LB'][:1],np.array([label]),b,components=comp)[0]['NLL']
    terms=[]
    for sd,lw,kernel in zip(comp[0],comp[1][0],comp[2]):
        kernel=np.asarray(kernel,float);norm=logsumexp(kernel)
        law=IntegerLaw(kernel-norm,b,tail='PG',signal=float(sd),K=K,g=G,sigma=RN)
        terms.append(lw+norm+law.logpmf(label))
    norm=logsumexp(comp[1][0]+logsumexp(np.asarray(comp[2],float),axis=1))
    de=abs(got+logsumexp(terms)-norm);assert de<1e-8
    return dict(status='PASS',width3_GPU_parity=err,width3_frozen_CPU_parity=err2,
        shallow_metric_parity=True,deep_mixture_integer_NLL_error=float(de),deep_stress_label=label,
        estimator_changed=False,finite_window_choices=[1,3,5,9,15])
