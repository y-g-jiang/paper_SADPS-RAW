import numpy as np
import torch
from scipy.special import logsumexp,log_ndtr
from .common import G,RN,K
from space_v2.cpu_pg import adaptive
from space_v2.integer_scoring import IntegerLaw

EVENTS=[('zero',0),('negative',1),('le_minus2',2),('le_minus3',3),('le_minus4',4),('le_minus6',6),('le_minus8',8)]

def arrays(logq,teacher,v,b,signal=None):
    q=np.asarray(logq,float);p=np.asarray(teacher,float);v=np.asarray(v)
    assert q.shape==p.shape and q.shape[1]==K+2 and np.isfinite(q).all() and np.isfinite(p).all()
    b=np.broadcast_to(np.asarray(b,float),v.shape)
    q=q-logsumexp(q,axis=1)[:,None];p=p-logsumexp(p,axis=1)[:,None]
    Q=np.exp(q);P=np.exp(p);w=-v.astype(int)
    assert (v<=0).all() and (v==np.floor(v)).all()


    lr=log_ndtr((-K-1.5-b)/RN)-log_ndtr((-K-.5-b)/RN)
    den=-np.expm1(lr);worst=np.maximum(Q[:,-1],P[:,-1])
    bound=worst*np.exp(lr)*((2*K+3)/den+2*np.exp(lr)/den**2)
    assert bound.max()<1e-12,'Use exact full-support moments for a wider tail'
    codes=-np.arange(K+2,dtype=float);mu=Q@codes;omu=P@codes
    qc=Q.cumsum(1);pc=P.cumsum(1);idx=np.minimum(w,K+1)
    nll=-q[np.arange(len(v)),idx]
    crps=((qc[:,:K+1]-(np.arange(K+1)[None]>=w[:,None]))**2).sum(1)+np.maximum(w-K-1,0)
    deep=w>K
    if deep.any():
        assert signal is not None,'Deep observed labels need component-resolved PG signal'
        for i in np.flatnonzero(deep):
            law=IntegerLaw(q[i],float(b[i]),tail='PG',signal=float(signal[i]),K=K,g=G,sigma=RN)
            nll[i]=-law.logpmf(int(v[i]));crps[i]=law.score(int(v[i]))['CRPS_integer']
    lo=(qc<.05).sum(1);hi=(qc<.95).sum(1)
    values=dict(KL=np.sum(P*(p-q),1),J_CDF=((pc-qc)**2).sum(1),NLL=nll,CRPS=crps,
        mean_MSE=(mu-v)**2,oracle_mean_MSE=(omu-v)**2,oracle_variance=P@(codes**2)-omu**2,
        oracle_mean_gap=(mu-omu)**2,coverage90=((w>=lo)&(w<=hi)).astype(float),zero_atom_error=Q[:,0]-P[:,0])
    calibration=[]
    for name,t in EVENTS:
        prob=Q[:,0] if t==0 else Q[:,t:].sum(1);target=(w==0) if t==0 else (w>=t)
        for j in range(10):
            mask=np.minimum((prob*10).astype(int),9)==j
            if mask.any():calibration.append(dict(event=name,bin=j,n=int(mask.sum()),prediction=float(prob[mask].sum()),observed=int(target[mask].sum())))
    return dict(n=len(v),**{k:float(x.mean()) for k,x in values.items()}),calibration,float(bound.max())

def aggregate(rows):
    if not rows:return dict(n=0,mothers=0)
    groups={}
    for r in rows:groups.setdefault(r['scene'],[]).append(r)
    keys=['KL','J_CDF','NLL','CRPS','mean_MSE','oracle_mean_MSE','oracle_variance','oracle_mean_gap','coverage90','zero_atom_error']
    total=sum(r['n'] for r in rows)
    return dict(n=total,mothers=len(groups),mother_equal={k:float(np.mean([np.mean([r[k] for r in rr]) for rr in groups.values()])) for k in keys},
                pixel_weighted={k:float(sum(r[k]*r['n'] for r in rows)/total) for k in keys})
