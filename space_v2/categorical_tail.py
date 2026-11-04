import numpy as np
from scipy.special import log_ndtr
from .common import RN

POLICY='categorical_fixed_zero_electron_quantized_gaussian_tail_v1'

def _validate(b,sigma,K):
    b=np.asarray(b,dtype=np.float64)
    if not np.all(np.isfinite(b)) or not np.isfinite(sigma) or sigma<=0 or int(K)!=K or K<0:
        raise ValueError('Finite bias, positive sigma and integer K>=0 required')
    return b

def conditional_moments(b,sigma=RN,K=64,tolerance=1e-14,max_terms=10000):






    b=_validate(b,sigma,K); first=K+1
    logden=log_ndtr((-K-.5-b)/sigma)
    mean_w=np.zeros_like(b); second_w=np.zeros_like(b)
    for j in range(1,max_terms+1):
        log_s=log_ndtr((-K-j-.5-b)/sigma)-logden
        log_next=log_ndtr((-K-j-1.5-b)/sigma)-logden
        sj=np.exp(log_s); nxt=np.exp(log_next)
        ratio=np.exp(log_next-log_s); den=-np.expm1(log_next-log_s)
        mean_w+=sj; second_w+=(2*j-1)*sj
        bound_mean=nxt/den
        bound_second_w=nxt*((2*j+1)/den+2*ratio/den**2)
        bound_second_v=2*first*bound_mean+bound_second_w
        if np.all(bound_second_v<tolerance):break
    else:raise RuntimeError('Tail moment series did not meet its error bound')
    mean=-first-mean_w; second=first**2+2*first*mean_w+second_w
    return dict(mean=mean,second=second,variance=second_w-mean_w**2,
        omitted_abs_mean_bound=bound_mean,omitted_second_bound=bound_second_v,terms=j)

def conditional_sample(u,b,sigma=RN,K=64,max_magnitude=32768):

    b=_validate(b,sigma,K);u,b=np.broadcast_arrays(np.asarray(u,np.float64),b)
    if np.any(~np.isfinite(u)) or np.any((u<=0)|(u>=1)):raise ValueError('u must lie strictly between zero and one')
    den=log_ndtr((-K-.5-b)/sigma); target=np.log1p(-u)
    magnitude=np.full(u.shape,K+1,dtype=np.int64)
    for _ in range(max_magnitude-K):
        log_remaining=log_ndtr((-magnitude-.5-b)/sigma)-den
        pending=log_remaining>target
        if not np.any(pending):return -magnitude
        magnitude[pending]+=1
        if np.any(magnitude>max_magnitude):raise OverflowError('Tail draw exceeds signed integer support')
    raise OverflowError('Tail inverse did not terminate')

def completed_moments(q,b,sigma=RN,K=64):
    q=np.asarray(q,np.float64)
    if q.shape[-1]!=K+2 or np.any(~np.isfinite(q)) or np.any(q<0) or np.any(abs(q.sum(-1)-1)>1e-10):
        raise ValueError('Expected normalized nonnegative coarsened PMF')
    tail=conditional_moments(b,sigma,K);codes=-np.arange(K+1,dtype=np.float64)
    mean=q[...,:K+1]@codes+q[...,-1]*tail['mean']
    second=q[...,:K+1]@(codes**2)+q[...,-1]*tail['second']

    variance=(q[...,:K+1]*(codes-mean[...,None])**2).sum(-1)+q[...,-1]*(tail['variance']+(tail['mean']-mean)**2)
    return dict(mean=mean,second=second,variance=variance,
        mean_error_bound=q[...,-1]*tail['omitted_abs_mean_bound'],second_error_bound=q[...,-1]*tail['omitted_second_bound'])

def sample_coarsened(q,u_bin,u_tail,b,sigma=RN,K=64):





    q=np.asarray(q,np.float64)
    completed_moments(q,b,sigma,K)
    u_bin=np.broadcast_to(np.asarray(u_bin,np.float64),q.shape[:-1])
    u_tail=np.broadcast_to(np.asarray(u_tail,np.float64),q.shape[:-1])
    if np.any(~np.isfinite(u_bin)) or np.any((u_bin<=0)|(u_bin>=1)):
        raise ValueError('Bucket uniforms must lie strictly between zero and one')
    if np.any(~np.isfinite(u_tail)) or np.any((u_tail<=0)|(u_tail>=1)):
        raise ValueError('Tail uniforms must lie strictly between zero and one')
    cdf=np.cumsum(q,axis=-1);cdf[...,-1]=1
    bucket=(u_bin[...,None]>cdf).sum(-1)
    values=np.asarray(-bucket,dtype=np.int64)
    tail=bucket==K+1
    if np.any(tail):
        biases=np.broadcast_to(np.asarray(b,np.float64),q.shape[:-1])
        values[tail]=conditional_sample(u_tail[tail],biases[tail],sigma,K)
    return values
