"""Portable physics/sampling smoke; no model, data download, or project imports.

python physics/smoke.py --output smoke
This checks the measurement contract, NOT N0 performance.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,log_ndtr,gammaln
from scipy.stats import poisson

G=21.601092412099675;SIGMA=24.95960511926749;K=304

def law(signal,bias=0):
    lam=signal/G;n=np.arange(int(poisson.ppf(1-1e-14,lam))+1 if lam else 1)
    lp=-lam+n*np.log(max(lam,1e-300))-gammaln(n+1)
    mu=G*n+bias;den=logsumexp(lp+log_ndtr((.5-mu)/SIGMA))
    k=-np.arange(K+1)[:,None]
    hi=log_ndtr((k+.5-mu)/SIGMA);lo=log_ndtr((k-.5-mu)/SIGMA)
    q=logsumexp(lp+hi+np.log(-np.expm1(lo-hi)),axis=1)-den
    tail=logsumexp(lp+log_ndtr((-K-.5-mu)/SIGMA))-den
    q=np.r_[q,tail];q-=logsumexp(q);return q

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(7103)
    yy,xx=np.mgrid[:64,:64];signal=(xx+yy)/128*24
    z=G*rng.poisson(signal/G)+rng.normal(0,SIGMA,signal.shape)
    adc=np.floor(512+z+.5);assert adc.min()>=0
    v=(np.minimum(adc,16383)-512).astype(np.int16);y=np.maximum(v,0);signed=y.copy();mask=y==0
    for r,c in np.argwhere(mask):
        q=law(float(signal[r,c]));cdf=np.exp(q).cumsum();j=int(np.searchsorted(cdf,rng.random()))
        if j>K:raise RuntimeError('Deep tail reached: use packaged project sampler for its exact continuation; never collapse tail to -305')
        signed[r,c]=-j
    assert np.array_equal(signed[y>0],y[y>0]) and np.array_equal(np.maximum(signed,0),y)
    normal=max(abs(np.exp(law(s)).sum()-1) for s in [0,1,16,64])
    result=dict(status='PASS',seed=7103,shape=[64,64],gain=G,analog_RN=SIGMA,quantizer='floor(512+Z+.5)',candidates=int(mask.sum()),negative_samples=int((signed<0).sum()),
        positive_identity_violations=0,reclip_violations=0,max_normalization_error=float(normal),N0_evaluated=False,
        qualification='Small generator/oracle-sampler contract smoke only; tail is rejected if reached, never replaced by a point mass.')
    (out/'SMOKE_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'SMOKE_ARRAYS.npz',observed=y,hidden_signed=v,sampled_signed=signed)
    print(json.dumps(result))

if __name__=='__main__':main()
