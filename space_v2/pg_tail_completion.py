import numpy as np
from scipy.special import log_ndtr
from .common import G, RN
from .cpu_pg import adaptive


def conditional_moments(signal, b, K=64, g=G, sigma=RN, tolerance=1e-13, max_terms=1000):
    s, b = np.broadcast_arrays(np.asarray(signal, float), np.asarray(b, float))
    if np.any(~np.isfinite(s)) or np.any(s < 0) or np.any(~np.isfinite(b)):
        raise ValueError('Finite nonnegative signal and finite bias required')
    if not np.isfinite(g) or g <= 0 or not np.isfinite(sigma) or sigma <= 0 or int(K) != K or K < 0:
        raise ValueError('Positive physical parameters and nonnegative integer cutoff required')

    means=[]; seconds=[]; variances=[]; bounds1=[]; bounds2=[]; terms=[]
    for signal_i, bias in zip(s.ravel(), b.ravel()):
        lam=float(signal_i/g); bias=float(bias); first=K+1
        logden=float(adaptive(lam,-K-.5,bias,g,sigma,'cdf')[0])
        mean_w=0.; second_w=0.
        for j in range(1,max_terms+1):
            log_s=float(adaptive(lam,-K-j-.5,bias,g,sigma,'cdf')[0])-logden
            survival=float(np.exp(log_s))
            mean_w+=survival; second_w+=(2*j-1)*survival
            log_ratio=float(log_ndtr((-K-j-1.5-bias)/sigma)-log_ndtr((-K-j-.5-bias)/sigma))
            ratio=float(np.exp(log_ratio)); den=float(-np.expm1(log_ratio))
            next_bound=survival*ratio
            bound_mean=next_bound/den
            bound_second_w=next_bound*((2*j+1)/den+2*ratio/den**2)
            bound_second=2*first*bound_mean+bound_second_w
            if bound_second<tolerance:break
        else:raise RuntimeError('PG tail moment bound not reached')
        means.append(-first-mean_w); seconds.append(first**2+2*first*mean_w+second_w)
        variances.append(second_w-mean_w**2); bounds1.append(bound_mean); bounds2.append(bound_second);terms.append(j)
    def shaped(values):return np.array(values).reshape(s.shape)
    return dict(mean=shaped(means),second=shaped(seconds),variance=shaped(variances),
        omitted_abs_mean_bound=shaped(bounds1),omitted_second_bound=shaped(bounds2),terms=shaped(terms))
