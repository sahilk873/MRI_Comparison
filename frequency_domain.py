"""
Frequency-domain summaries for ADC slices (manuscript Section 3.11).

- Spectral slope: log-log linear fit over the full radial profile (excluding r=0).
- High-frequency fraction: proportion of total power at radii beyond 35% of the
  maximum radial extent (cumulative fraction of power in the high-frequency tail).
"""

import numpy as np
from typing import Tuple, Optional


def radial_power_spectrum_1d(slice_2d: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute radially averaged 1D power spectrum from a 2D slice.

    Parameters
    ----------
    slice_2d : np.ndarray, shape (H, W)
        Mid-axial slice (e.g. after robust scaling).
    mask : np.ndarray, shape (H, W), optional
        If provided, mask applied before FFT (zeros outside mask).

    Returns
    -------
    np.ndarray
        1D array of mean power at each radial bin (length ~ min(H,W)//2).
    """
    if mask is not None:
        slice_2d = np.where(mask, slice_2d, 0.0)
    f = np.fft.fft2(slice_2d)
    f = np.fft.fftshift(f)
    power = np.abs(f) ** 2
    H, W = power.shape
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(yy.astype(float) ** 2 + xx.astype(float) ** 2).astype(int)
    max_r = min(cy, cx)
    radial_sum = np.zeros(max_r + 1)
    radial_count = np.zeros(max_r + 1, dtype=int)
    for ri in range(max_r + 1):
        sel = r == ri
        radial_sum[ri] = np.sum(power[sel])
        radial_count[ri] = np.sum(sel)
    radial_count[0] = max(radial_count[0], 1)
    profile = radial_sum / radial_count
    return profile


def spectral_slope_and_hf_fraction(
    slice_2d: np.ndarray,
    mask: Optional[np.ndarray] = None,
    hf_radius_percentile: float = 35.0,
) -> Tuple[float, float]:
    """
    Spectral slope (log-log fit, excluding r=0) and high-frequency energy fraction.

    Parameters
    ----------
    slice_2d : np.ndarray
        Mid-axial slice; will be robustly scaled to [0,1] within mask if provided.
    mask : np.ndarray, optional
        Mask for valid pixels.
    hf_radius_percentile : float
        High-frequency band: radii beyond this fraction of max radius (default 35%).

    Returns
    -------
    slope : float
        Slope of log(power) vs log(radius) (excluding r=0).
    hf_fraction : float
        Proportion of total power at radii beyond hf_radius_percentile of max.
    """
    if mask is not None:
        vals = slice_2d[mask]
    else:
        vals = slice_2d.ravel()
    p1, p99 = np.percentile(vals, [1, 99])
    slice_scaled = np.clip((slice_2d - p1) / (p99 - p1 + 1e-10), 0, 1).astype(np.float64)

    profile = radial_power_spectrum_1d(slice_scaled, mask)
    n = len(profile)
    if n < 3:
        return np.nan, np.nan
    # Exclude r=0
    r = np.arange(1, n, dtype=float)
    p = profile[1:n]
    valid = p > 0
    if np.sum(valid) < 2:
        return np.nan, np.nan
    log_r = np.log(r[valid])
    log_p = np.log(p[valid])
    slope = np.polyfit(log_r, log_p, 1)[0]

    # High-frequency fraction: power at radii > hf_radius_percentile of max
    max_radius_idx = n - 1
    thresh_idx = max(1, int(max_radius_idx * (hf_radius_percentile / 100.0)))
    total_power = np.sum(profile[1:])
    if total_power <= 0:
        return slope, np.nan
    hf_power = np.sum(profile[thresh_idx:])
    hf_fraction = hf_power / total_power
    return float(slope), float(hf_fraction)
