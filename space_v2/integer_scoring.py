from functools import lru_cache
import numpy as np
from scipy.special import log_ndtr, logsumexp
from physics.pg_quantized import normal_interval_logprob
from .common import G, RN
from .cpu_pg import adaptive
from .categorical_tail import conditional_moments as c_moments
from .pg_tail_completion import conditional_moments as pg_moments

SCHEMA='SPACE_V2_completed_integer_reference_v1'


class IntegerLaw:
    def __init__(self,logq,b,tail='C',signal=None,K=64,g=G,sigma=RN):
        self.logq=np.asarray(logq,dtype=float).copy()
        if int(K)!=K or K<0 or self.logq.shape!=(K+2,) or np.any(np.isnan(self.logq)) or np.any(np.isposinf(self.logq)):
            raise ValueError('One valid coarsened log PMF is required')
        z=float(logsumexp(self.logq))
        if not np.isfinite(z) or abs(z)>1e-8:raise ValueError('Log PMF must be normalized')
        if tail not in ['C','PG'] or not np.isfinite(b) or not np.isfinite(g) or g<=0 or not np.isfinite(sigma) or sigma<=0:
            raise ValueError('Invalid physical parameters or tail law')
        if tail=='PG' and (signal is None or not np.isfinite(signal) or signal<0):raise ValueError('PG needs nonnegative predicted signal')
        self.logq-=z;self.q=np.exp(self.logq);self.K=int(K);self.b=float(b);self.tail=tail;self.signal=signal;self.g=g;self.sigma=sigma
        self.den=self.log_event(-K-.5)

    def log_event(self,threshold):
        if self.tail=='C':return float(log_ndtr((threshold-self.b)/self.sigma))
        return float(adaptive(self.signal/self.g,threshold,self.b,self.g,self.sigma,'cdf')[0])

    @lru_cache(maxsize=256)
    def log_survival(self,m):

        if m<1 or int(m)!=m:raise ValueError('Positive integer threshold required')
        if m<=self.K+1:return float(logsumexp(self.logq[int(m):]))
        return float(self.logq[-1]+self.log_event(-m+.5)-self.den)

    def logpmf(self,v):
        if not np.isfinite(v) or int(v)!=v or v>0:raise ValueError('Nonpositive integer signed label required')
        n=-int(v)
        if n<=self.K:return float(self.logq[n])
        lp=(float(normal_interval_logprob(v-.5-self.b,v+.5-self.b,self.sigma)) if self.tail=='C'
            else float(adaptive(self.signal/self.g,v,self.b,self.g,self.sigma,'bin')[0]))
        return float(self.logq[-1]+lp-self.den)

    def moments(self):
        codes=-np.arange(self.K+1,dtype=float)
        if self.q[-1]>0:
            t=(c_moments(self.b,self.sigma,self.K) if self.tail=='C' else pg_moments(self.signal,self.b,self.K,self.g,self.sigma))
            tm=float(t['mean']);tv=float(t['variance']);ts=float(t['second'])
        else:tm=tv=ts=0.
        mean=float(self.q[:-1]@codes+self.q[-1]*tm)
        second=float(self.q[:-1]@(codes**2)+self.q[-1]*ts)
        variance=float(self.q[:-1]@((codes-mean)**2)+self.q[-1]*(tv+(tm-mean)**2))
        return dict(mean=mean,second=second,variance=variance)

    def magnitude_quantile(self,p,max_code=32768):
        if not 0<p<1:raise ValueError('Quantile level must be inside (0,1)')
        threshold=np.log1p(-p)
        for k in range(max_code+1):
            if self.log_survival(k+1)<=threshold:return k
        raise OverflowError('Quantile beyond signed support')

    def score(self,v,tolerance=1e-12):
        nll=-self.logpmf(v);w=-int(v)


        crps=sum((np.exp(self.log_survival(m))-float(w>=m))**2 for m in range(1,self.K+2))
        crps+=max(w-self.K-1,0)
        for m in range(self.K+2,self.K+10002):
            s=float(np.exp(self.log_survival(m)));crps+=s*s-2*s*(w>=m)

            lr=float(log_ndtr((-m-.5-self.b)/self.sigma)-log_ndtr((-m+.5-self.b)/self.sigma))
            r=float(np.exp(lr));next_s=s*r
            bound=next_s**2/(-np.expm1(2*lr))+2*next_s/(-np.expm1(lr))
            if bound<tolerance:break
        else:raise RuntimeError('Integer CRPS tail bound not reached')
        lo=self.magnitude_quantile(.05);hi=self.magnitude_quantile(.95);moments=self.moments()
        return dict(NLL_integer=float(nll),CRPS_integer=float(crps),CRPS_omitted_abs_bound=float(bound),
            interval90_signed=[-hi,-lo],coverage90_integer=bool(lo<=w<=hi),
            posterior_mean= moments['mean'],posterior_variance=moments['variance'],mean_MSE_integer=(moments['mean']-v)**2)


def paired_score(prediction,oracle,v):
    if prediction.K!=oracle.K:raise ValueError('Coarsened supports must match')
    p=oracle.q;positive=p>0
    kl=float(np.sum(p[positive]*(oracle.logq[positive]-prediction.logq[positive])))
    j=float(np.sum((p.cumsum()-prediction.q.cumsum())**2))
    a=prediction.score(v);b=oracle.score(v)
    return dict(schema=SCHEMA,KL_coarsened=kl,J_CDF_coarsened=j,**a,
        oracle_mean_MSE_integer=b['mean_MSE_integer'],oracle_expected_variance=b['posterior_variance'],
        realized_MSE_gap=a['mean_MSE_integer']-b['mean_MSE_integer'],
        posterior_mean_gap=(a['posterior_mean']-b['posterior_mean'])**2)
