import numpy as np
from scipy.special import logsumexp
from space_v2.categorical_sampling import coordinate_uniform
from space_v2.cpu_pg import adaptive
from .common import G,RN,K

def sample(logq,signal,b,ids,seed=7103,draw=0,image_id=12800):
    lq=np.asarray(logq,float);signal=np.asarray(signal);b=np.asarray(b);ids=np.asarray(ids,np.uint64)
    assert lq.shape==(len(ids),K+2)
    lq=lq-logsumexp(lq,axis=1)[:,None];cdf=np.exp(lq).cumsum(1);cdf[:,-1]=1
    u=coordinate_uniform(ids,seed,draw,image_id,0);index=(u[:,None]>cdf).sum(1)
    output=-index.astype(np.int16)
    for i in np.flatnonzero(index==K+1):
        ut=coordinate_uniform(ids[i:i+1],seed,draw,image_id,1)[0]
        den=float(adaptive(float(signal[i]/G),-K-.5,float(b[i]),G,RN,'cdf')[0])
        for m in range(K+1,32768):
            survival=float(adaptive(float(signal[i]/G),-m-.5,float(b[i]),G,RN,'cdf')[0])-den
            if survival<=np.log1p(-ut):output[i]=-m;break
        else:raise OverflowError('Signed tail exceeds int16')
    assert (output<=0).all()
    return output
