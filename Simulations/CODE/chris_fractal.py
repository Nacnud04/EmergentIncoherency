import numpy as np
import scipy.integrate as integrate

C = 3e8
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def lens_kernel(x):
    """
    Pair-separation kernel x A(x) for a hard Fresnel disk.
    """
    half = np.arccos(min(x / 2, 1)) / 2
    th = half * (GL_NODES + 1)
    return (
        2 / np.pi
        * half
        * np.sum(
            GL_WEIGHTS
            * np.sin(2 * np.pi * x * (np.cos(th) - x / 2))
            * np.cos(th)
        )
    )

def fresnel_radius(f, h, e=1.0):
    """
    First Fresnel zone radius sqrt(lambda h / 2).
    """
    return np.sqrt(C / (np.asarray(f) * np.sqrt(e)) * h / 2)


def disk_loss(sigma_l, H, h_l):
    sigma_l = float(np.asarray(sigma_l))
    h_l = float(np.asarray(h_l))

    a = (4 * np.pi)**2 * sigma_l**2 * (h_l / 2)**H
    integrand = lambda x: lens_kernel(x) * np.exp(-a * x**(2 * H))

    return np.pi / 2 * integrate.quad(
        integrand,
        0,
        2,
        points=np.geomspace(1e-6, 1, 13),
        epsabs=0,
        limit=200,
    )[0]


def fractal_loss(sigma, H, f, h, lag=None, e=1.0, allan=False):
    lam = C / (np.asarray(f) * np.sqrt(e))
    s = np.asarray(sigma) / np.sqrt(2) if allan else np.asarray(sigma)
    lag = lam if lag is None else lag

    h_arr = np.asarray(h)
    sigma_l = s * (lam / lag)**H / lam

    return np.array([disk_loss(sigma_l, H, hh / lam) for hh in np.atleast_1d(h_arr)])



def roughness_loss_report(sigma, H, f, h, lag=None, e=1.0, allan=False):
    """
    Convenience wrapper that returns the computed loss plus useful derived values.
    """
    lam = C / (np.asarray(f) * np.sqrt(e))
    lag = lam if lag is None else lag
    L = fractal_loss(sigma, H, f, h, lag=lag, e=e, allan=allan)
    sl = (sigma / np.sqrt(2) if allan else sigma) * (lam / lag) ** H

    return {
        "lambda_m": lam,
        "fresnel_radius_m": fresnel_radius(f, h, e),
        "lag_m": lag,
        "H": H,
        "sigma_input_m": sigma,
        "sigma_at_lambda_m": sl,
        "loss_linear": L,
        "loss_db": 10 * np.log10(L),
    }

