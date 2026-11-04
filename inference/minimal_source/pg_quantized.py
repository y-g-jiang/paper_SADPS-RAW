from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.special import gammaln, log_ndtr, ndtri
from scipy.stats import poisson


def poisson_n_max(lam: float | np.ndarray, tol: float = 1e-12, extra: int = 2) -> int:

    if tol <= 0 or tol >= 1:
        raise ValueError("tol must be in (0, 1)")
    lm = float(np.max(np.asarray(lam, dtype=np.float64)))
    if not np.isfinite(lm) or lm < 0:
        raise ValueError("lambda must be finite and non-negative")
    if lm == 0:
        return 0
    n = int(max(0, math.ceil(float(poisson.ppf(1.0 - tol, lm))))) + int(extra)

    while float(poisson.sf(n, lm)) > tol:
        n = max(n + 1, int(math.ceil(n * 1.05)))
    return n


def _logdiffexp(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    out = np.full(np.broadcast(a, b).shape, -np.inf, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    ok = a > b
    if np.any(ok):
        d = np.clip(b[ok] - a[ok], -745.0, 0.0)
        out[ok] = a[ok] + np.log1p(-np.exp(d))
    return out


def normal_interval_logprob(lower, upper, sigma):

    lower, upper, sigma = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64)
    )
    if np.any(sigma <= 0) or np.any(upper < lower):
        raise ValueError("invalid normal interval")
    lo = lower / sigma
    hi = upper / sigma

    out = np.empty_like(lo)
    left = hi <= 0
    right = lo >= 0
    middle = ~(left | right)
    out[left] = _logdiffexp(log_ndtr(hi[left]), log_ndtr(lo[left]))
    out[right] = _logdiffexp(log_ndtr(-lo[right]), log_ndtr(-hi[right]))
    out[middle] = _logdiffexp(log_ndtr(hi[middle]), log_ndtr(lo[middle]))
    return out


def _broadcast_params(lam, b, g, sigma, delta):
    lam, b, g, sigma, delta = np.broadcast_arrays(
        np.asarray(lam, dtype=np.float64), np.asarray(b, dtype=np.float64),
        np.asarray(g, dtype=np.float64), np.asarray(sigma, dtype=np.float64),
        np.asarray(delta, dtype=np.float64)
    )
    if np.any(lam < 0) or np.any(g <= 0) or np.any(sigma <= 0) or np.any(delta <= 0):
        raise ValueError("invalid PG parameters")
    return lam, b, g, sigma, delta


def _log_pois(n, lam):
    n = np.asarray(n, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -lam + n * np.log(lam) - gammaln(n + 1.0)
    return np.where((lam == 0) & (n == 0), 0.0, np.where(lam == 0, -np.inf, out))


def pg_logpmf(k, lam, b, g, sigma, delta=1.0, tol=1e-12):





    lam, b, g, sigma, delta = _broadcast_params(lam, b, g, sigma, delta)
    k = np.asarray(k, dtype=np.float64)
    if k.ndim and lam.ndim:
        if k.ndim != 1:
            raise ValueError("array k with array parameters is not supported in NumPy reference path")
        if lam.size != 1:
            raise ValueError("array k with array parameters is not supported in NumPy reference path")
        lam, b, g, sigma, delta = [x.reshape(()) for x in (lam, b, g, sigma, delta)]
    nmax = poisson_n_max(lam, tol=tol)
    n = np.arange(nmax + 1, dtype=np.float64)
    if k.ndim and lam.ndim == 0:

        lp = _log_pois(n, lam)
        lower = (k[..., None] - 0.5) * delta - b - g * n[None, :]
        upper = (k[..., None] + 0.5) * delta - b - g * n[None, :]
        log_interval = normal_interval_logprob(lower, upper, sigma)
        return np.logaddexp.reduce(lp[None, :] + log_interval, axis=-1)
    lp = _log_pois(n, lam[..., None])
    lower = (k[..., None] - 0.5) * delta[..., None] - b[..., None] - g[..., None] * n
    upper = (k[..., None] + 0.5) * delta[..., None] - b[..., None] - g[..., None] * n
    log_interval = normal_interval_logprob(lower, upper, sigma[..., None])
    return np.logaddexp.reduce(lp + log_interval, axis=-1)


def pg_zero_event_logprob(lam, b, g, sigma, delta=1.0, tol=1e-12):

    lam, b, g, sigma, delta = _broadcast_params(lam, b, g, sigma, delta)
    nmax = poisson_n_max(lam, tol=tol)
    n = np.arange(nmax + 1, dtype=np.float64)
    lp = _log_pois(n, lam[..., None])
    threshold = (0.5 * delta)[..., None] - b[..., None] - g[..., None] * n
    log_cdf = log_ndtr(threshold / sigma[..., None])
    return np.logaddexp.reduce(lp + log_cdf, axis=-1)


def choose_code_min(lam, b, g, sigma, delta=1.0, tail_sigma=10.0, floor=-8):

    mean = np.asarray(b) + np.asarray(g) * np.asarray(lam)
    std = np.sqrt(np.asarray(g) ** 2 * np.asarray(lam) + np.asarray(sigma) ** 2)
    d = np.asarray(delta)
    kmin = np.floor((mean - tail_sigma * std - 0.5 * d) / d).astype(int)
    kmin = np.minimum(kmin, -1)
    kmin = np.minimum(kmin, floor)
    return int(np.min(kmin))


def conditional_pmf(lam, b, g, sigma, delta=1.0, tol=1e-12, k_min=None):

    if np.asarray(lam).size != 1:
        raise ValueError("conditional_pmf reference path is scalar")
    lam, b, g, sigma, delta = [float(np.asarray(x)) for x in _broadcast_params(lam, b, g, sigma, delta)]
    if k_min is None:
        k_min = choose_code_min(lam, b, g, sigma, delta)
    if k_min >= 0:
        k_min = -1
    while True:
        codes = np.arange(k_min, 1, dtype=np.int64)
        log_mass = pg_logpmf(codes, lam, b, g, sigma, delta, tol=tol)
        log_zero = float(pg_zero_event_logprob(lam, b, g, sigma, delta, tol=tol))
        probs = np.exp(log_mass - log_zero)
        if np.isfinite(probs).all() and probs.sum() >= 1.0 - max(1e-10, 10 * tol):
            probs /= probs.sum()
            return codes, probs, float(np.exp(log_zero))
        if k_min < -8192:
            raise RuntimeError("failed to capture lower-tail posterior mass")
        k_min *= 2


def conditional_logprob(k, lam, b, g, sigma, delta=1.0, tol=1e-12):

    k = np.asarray(k, dtype=np.int64)
    if np.any(k > 0):
        raise ValueError("conditional code must be <= 0")
    return pg_logpmf(k, lam, b, g, sigma, delta, tol=tol) - pg_zero_event_logprob(lam, b, g, sigma, delta, tol=tol)


def sample_conditional(lam, b, g, sigma, delta=1.0, size=None, rng=None, tol=1e-12):

    rng = np.random.default_rng() if rng is None else rng
    lam, b, g, sigma, delta = [float(np.asarray(x)) for x in _broadcast_params(lam, b, g, sigma, delta)]
    nmax = poisson_n_max(lam, tol=tol)
    n = np.arange(nmax + 1, dtype=np.int64)
    logw = _log_pois(n, lam)
    threshold = (0.5 * delta - b - g * n) / sigma
    logw = logw + log_ndtr(threshold)
    w = np.exp(logw - np.logaddexp.reduce(logw))
    shape = () if size is None else (size,) if isinstance(size, int) else tuple(size)
    ns = rng.choice(n, size=shape, p=w)
    u = rng.random(shape)
    lim = ndtri(0.5 * 0 + np.exp(log_ndtr((0.5 * delta - b - g * ns) / sigma)))

    z = ndtri(np.clip(u * np.exp(log_ndtr((0.5 * delta - b - g * ns) / sigma)), 1e-15, 1 - 1e-15))
    r = sigma * z
    code = np.floor((b + g * ns + r) / delta + 0.5).astype(np.int64)
    return np.minimum(code, 0)


def gaussian_zero_reference(sigma):

    sigma = float(sigma)
    p_y0 = float(np.exp(log_ndtr(0.5 / sigma)))
    p_v0 = float((np.exp(log_ndtr(0.5 / sigma)) - np.exp(log_ndtr(-0.5 / sigma))) / p_y0)
    return p_y0, p_v0




def pg_conditional_pmf_torch(lam, b, g, sigma, delta, k_min=-64, n_max=None):
    import torch
    if not all(torch.is_tensor(x) for x in (lam, b, g, sigma, delta)):
        raise TypeError("Torch tensors required")
    lam = lam.clamp_min(0.0)
    if n_max is None:
        lm = float(lam.detach().max().cpu())
        n_max = max(32, poisson_n_max(lm, tol=1e-10))
    n = torch.arange(n_max + 1, device=lam.device, dtype=lam.dtype)
    nv = n.view(1, 1, 1, 1, -1)
    log_pois = -lam.unsqueeze(-1) + nv * torch.log(lam.clamp_min(1e-30)).unsqueeze(-1) - torch.lgamma(nv + 1)
    log_pois[..., 0] = torch.where(lam == 0, torch.zeros_like(lam), log_pois[..., 0])
    ks = torch.arange(k_min, 1, device=lam.device, dtype=lam.dtype)
    d = delta.unsqueeze(-1).unsqueeze(-1)
    bb = b.unsqueeze(-1).unsqueeze(-1)
    gg = g.unsqueeze(-1).unsqueeze(-1)
    ss = sigma.unsqueeze(-1).unsqueeze(-1)

    masses=[]
    for ka in range(0, int(ks.numel()), 8):
        kc = ks[ka:ka+8]
        kv = kc.view(1, 1, 1, -1, 1)
        lower = (kv - 0.5) * d - bb - gg * nv
        upper = (kv + 0.5) * d - bb - gg * nv
        log_phi_u = torch.special.log_ndtr(upper / ss)
        log_phi_l = torch.special.log_ndtr(lower / ss)
        a = torch.maximum(log_phi_u, log_phi_l)
        diff = torch.minimum(log_phi_u, log_phi_l)
        log_interval = a + torch.log((-torch.expm1(diff - a)).clamp_min(1e-30))
        masses.append(torch.logsumexp(log_pois.unsqueeze(-2) + log_interval, dim=-1))
        del lower, upper, log_phi_u, log_phi_l, log_interval
    log_mass = torch.cat(masses, dim=-1)
    zero_upper = 0.5 * delta.unsqueeze(-1) - b.unsqueeze(-1) - g.unsqueeze(-1) * n
    log_zero = torch.logsumexp(log_pois.squeeze(-2) + torch.special.log_ndtr(zero_upper / sigma.unsqueeze(-1)), dim=-1)
    log_q = log_mass - log_zero.unsqueeze(-1)
    q = torch.softmax(log_q, dim=-1)
    return ks, q, log_zero




def conditional_moments(lam, b, g, sigma, delta=1.0, tol=1e-12, k_min=None):

    codes, probs, p_zero = conditional_pmf(lam, b, g, sigma, delta, tol=tol, k_min=k_min)
    values = codes.astype(np.float64) * float(delta)
    mean = float(np.sum(probs * values))
    second = float(np.sum(probs * values * values))
    return mean, max(0.0, second - mean * mean), codes, probs, p_zero


def pg_conditional_logprob_torch(lam, b, g, sigma, delta, target, tol=1e-10):

    import torch
    lam=lam.clamp_min(0.0)
    lm=float(lam.detach().max().cpu()); n_max=max(32,poisson_n_max(lm,tol=tol))
    n=torch.arange(n_max+1,device=lam.device,dtype=lam.dtype).view(1,1,1,1,-1)
    log_pois=-lam.unsqueeze(-1)+n*torch.log(lam.clamp_min(1e-30)).unsqueeze(-1)-torch.lgamma(n+1)
    log_pois[...,0]=torch.where(lam==0,torch.zeros_like(lam),log_pois[...,0])
    k=torch.round(target/delta).clamp_max(0).to(lam.dtype).unsqueeze(-1)
    dd=delta.unsqueeze(-1); bb=b.unsqueeze(-1); gg=g.unsqueeze(-1); ss=sigma.unsqueeze(-1)
    lower=(k-0.5)*dd-bb-gg*n; upper=(k+0.5)*dd-bb-gg*n
    lu=torch.special.log_ndtr(upper/ss); ll=torch.special.log_ndtr(lower/ss); a=torch.maximum(lu,ll); d=torch.minimum(lu,ll)
    log_mass=torch.logsumexp(log_pois+a+torch.log((-torch.expm1(d-a)).clamp_min(1e-30)),dim=-1)
    zu=0.5*delta.unsqueeze(-1)-b.unsqueeze(-1)-g.unsqueeze(-1)*torch.arange(n_max+1,device=lam.device,dtype=lam.dtype)
    log_zero=torch.logsumexp(log_pois.squeeze(-2)+torch.special.log_ndtr(zu/sigma.unsqueeze(-1)),dim=-1)
    return log_mass-log_zero
