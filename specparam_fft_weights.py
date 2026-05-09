"""Bridge SpecParam frequency-domain model fits back to the time domain.

Converts a fitted PSD model (from SpecParam / FOOOF) into per-bin amplitude
weights on a signal's complex FFT. Multiplying the FFT magnitudes by these
weights (phases untouched) and inverting yields time-domain reconstructions
and residuals for any model component (aperiodic, periodic, or full).

Core functions (no specparam dependency):
    model_psd_to_weights  — PSD arrays -> amplitude-ratio weight vector
    apply_spectral_weights — weight vector + signal -> reconstructed signal
    extract_residual       — original - reconstruction

SpecParam integration (requires ``specparam`` package):
    weights_from_specparam  — fitted SpectralModel -> weight vector
    specparam_reconstruct   — fitted SpectralModel + signal -> (reconstruction, residual)

Usage example
-------------
>>> import numpy as np
>>> from scipy.signal import welch
>>> from specparam import SpectralModel
>>> from specparam_fft_weights import specparam_reconstruct
>>>
>>> sfreq = 256.0
>>> signal = np.random.default_rng(0).standard_normal(int(sfreq * 4))
>>> freqs, psd = welch(signal, fs=sfreq, nperseg=512)
>>> fm = SpectralModel(verbose=False)
>>> fm.fit(freqs, psd, [1, 50])
>>>
>>> # Extract aperiodic reconstruction and oscillatory residual
>>> result = specparam_reconstruct(signal, sfreq, fm, component='aperiodic')
>>> aperiodic_signal = result.reconstruction
>>> oscillatory_residual = result.residual
>>>
>>> # Or extract periodic (peaks-only) component
>>> result_p = specparam_reconstruct(signal, sfreq, fm, component='periodic')

Limitations
-----------
- Assumes the signal is approximately stationary over the analysis window.
  Non-stationary signals (transient artifacts, event-related responses) will
  produce decompositions that reflect the average spectral content, not
  time-varying dynamics.
- The model PSD comes from a Welch estimate (smoothed, averaged), while the
  reconstruction operates on the raw periodogram of the signal. The weights
  reshape the signal's spectrum toward the smooth model fit, which is usually
  desirable but means the reconstruction's PSD will differ from the raw
  periodogram.
- Decomposition quality is bounded by SpecParam's model accuracy. Poor fits
  (low r-squared, missed peaks, wrong aperiodic mode) propagate directly
  into the time-domain reconstruction.
- Frequency bins outside the SpecParam fit range (e.g., below 1 Hz or above
  the upper fit limit) are handled by the ``out_of_range`` parameter:
  'passthrough' (default, weight=1.0) or 'zero' (weight=0.0). Neither is
  a true model of out-of-range content.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

Component = Literal["full", "aperiodic", "periodic"]
OutOfRangePolicy = Literal["passthrough", "zero"]


def model_psd_to_weights(
    model_psd: np.ndarray,
    empirical_psd: np.ndarray,
    eps: float = 1e-20,
    max_weight: float = 100.0,
) -> np.ndarray:
    """Compute per-bin amplitude-ratio weights from model and empirical PSDs.

    Parameters
    ----------
    model_psd : np.ndarray, shape (n_freqs,)
        Model PSD in linear power (not log10).
    empirical_psd : np.ndarray, shape (n_freqs,)
        Empirical PSD in linear power (not log10). Must match model_psd shape.
    eps : float
        Floor applied to empirical_psd before division to avoid division by
        zero. Default 1e-20.
    max_weight : float
        Upper clamp on weights to prevent noise amplification. Default 100.0.

    Returns
    -------
    weights : np.ndarray, shape (n_freqs,)
        Amplitude-ratio weights, each in [0, max_weight].
        weight[i] = sqrt(model_psd[i] / empirical_psd[i]), clamped.
    """
    model_psd = np.asarray(model_psd, dtype=np.float64)
    empirical_psd = np.asarray(empirical_psd, dtype=np.float64)
    if model_psd.shape != empirical_psd.shape:
        raise ValueError(
            f"Shape mismatch: model_psd {model_psd.shape} vs "
            f"empirical_psd {empirical_psd.shape}"
        )

    empirical_safe = np.maximum(empirical_psd, eps)
    model_safe = np.maximum(model_psd, 0.0)
    weights = np.sqrt(model_safe / empirical_safe)
    np.clip(weights, 0.0, max_weight, out=weights)

    if not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite weights detected after computation")

    return weights


def apply_spectral_weights(
    signal: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply amplitude weights to a signal's FFT and invert back to time domain.

    Computes rfft(signal), multiplies magnitudes by weights (preserving
    complex phases), then irfft's back.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Real-valued time-domain signal.
    weights : np.ndarray, shape (n_freqs,)
        Per-bin amplitude weights. Length must equal n_samples // 2 + 1
        (the rfft output length).

    Returns
    -------
    reconstruction : np.ndarray, shape (n_samples,)
        Time-domain signal reconstructed from the weighted FFT.
    """
    signal = np.asarray(signal, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(signal)
    expected_n_freqs = n // 2 + 1
    if len(weights) != expected_n_freqs:
        raise ValueError(
            f"weights length {len(weights)} does not match expected "
            f"rfft length {expected_n_freqs} for signal of length {n}"
        )

    fft_coeffs = np.fft.rfft(signal)
    weighted_fft = fft_coeffs * weights
    return np.fft.irfft(weighted_fft, n=n)


def extract_residual(
    signal: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """Subtract reconstruction from original signal to get the residual.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Original time-domain signal.
    reconstruction : np.ndarray, shape (n_samples,)
        Model reconstruction of the signal.

    Returns
    -------
    residual : np.ndarray, shape (n_samples,)
        signal - reconstruction.
    """
    signal = np.asarray(signal, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    if signal.shape != reconstruction.shape:
        raise ValueError(
            f"Shape mismatch: signal {signal.shape} vs "
            f"reconstruction {reconstruction.shape}"
        )
    return signal - reconstruction


class ReconstructionResult(NamedTuple):
    """Result of specparam_reconstruct: time-domain reconstruction and residual."""
    reconstruction: np.ndarray
    residual: np.ndarray


def _get_model_spectrum_log10(
    model, component: Component, eps: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a component spectrum in log10 power from a fitted SpectralModel.

    Parameters
    ----------
    model : SpectralModel
        A fitted specparam model.
    component : {'full', 'aperiodic', 'periodic'}
        Which model component to extract.
    eps : float
        Floor for near-zero periodic power values before log10.

    Returns
    -------
    freqs : np.ndarray
        Frequency vector from the model.
    spectrum_log10 : np.ndarray
        Component spectrum in log10 power.
    """
    freqs = model.data.freqs
    if component == "full":
        return freqs, model.results.model.modeled_spectrum
    elif component == "aperiodic":
        return freqs, model.results.model.get_component("aperiodic")
    elif component == "periodic":
        full_linear = np.power(10, model.results.model.modeled_spectrum)
        ap_linear = np.power(10, model.results.model.get_component("aperiodic"))
        periodic_linear = np.maximum(full_linear - ap_linear, eps)
        return freqs, np.log10(periodic_linear)
    else:
        raise ValueError(
            f"Unknown component '{component}'. Use 'full', 'aperiodic', or 'periodic'."
        )


def _interpolate_model_to_fft_grid(
    model_freqs: np.ndarray,
    model_log10: np.ndarray,
    fft_freqs: np.ndarray,
    out_of_range: OutOfRangePolicy = "passthrough",
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate model PSD (log10) onto the signal's FFT frequency grid.

    Parameters
    ----------
    model_freqs : np.ndarray
        Frequency vector from the model fit.
    model_log10 : np.ndarray
        Model PSD in log10 power on model_freqs.
    fft_freqs : np.ndarray
        Target FFT frequency grid (from rfftfreq).
    out_of_range : {'passthrough', 'zero'}
        Policy for FFT bins outside the model frequency range.

    Returns
    -------
    model_linear : np.ndarray
        Model PSD in linear power on fft_freqs. Out-of-range bins are NaN
        (passthrough) or 0.0 (zero).
    out_of_range_mask : np.ndarray
        Boolean mask, True for bins outside the model frequency range.
    """
    model_log10_interp = np.interp(
        fft_freqs, model_freqs, model_log10,
        left=np.nan, right=np.nan,
    )
    out_of_range_mask = np.isnan(model_log10_interp)
    model_linear = np.power(10, model_log10_interp)

    if out_of_range == "zero":
        model_linear[out_of_range_mask] = 0.0
    elif out_of_range != "passthrough":
        raise ValueError(
            f"Unknown out_of_range policy '{out_of_range}'. "
            "Use 'passthrough' or 'zero'."
        )

    return model_linear, out_of_range_mask


def _compute_weights_and_fft(
    model,
    signal: np.ndarray,
    sfreq: float,
    component: Component = "full",
    out_of_range: OutOfRangePolicy = "passthrough",
    eps: float = 1e-20,
    max_weight: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weights and return the signal's FFT coefficients.

    Shared implementation for weights_from_specparam and specparam_reconstruct
    so the FFT is computed only once.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    fft_coeffs = np.fft.rfft(signal)
    empirical_psd = fft_coeffs.real ** 2 + fft_coeffs.imag ** 2

    model_freqs, model_log10 = _get_model_spectrum_log10(model, component, eps=eps)
    model_linear, oor_mask = _interpolate_model_to_fft_grid(
        model_freqs, model_log10, fft_freqs, out_of_range,
    )

    # Welch PSD and raw periodogram have different normalizations.
    # Scale the model to match the periodogram's total in-range power.
    in_range = ~oor_mask
    if np.any(in_range):
        safe_model_sum = max(np.sum(model_linear[in_range]), eps)
        model_linear *= np.sum(empirical_psd[in_range]) / safe_model_sum

    if out_of_range == "passthrough":
        model_for_weights = np.where(oor_mask, empirical_psd, model_linear)
    else:
        model_for_weights = np.where(oor_mask, 0.0, model_linear)

    weights = model_psd_to_weights(
        model_for_weights, empirical_psd, eps=eps, max_weight=max_weight,
    )
    return weights, fft_coeffs


def weights_from_specparam(
    model,
    signal: np.ndarray,
    sfreq: float,
    component: Component = "full",
    out_of_range: OutOfRangePolicy = "passthrough",
    eps: float = 1e-20,
    max_weight: float = 100.0,
) -> np.ndarray:
    """Compute FFT amplitude weights from a fitted SpecParam model.

    Parameters
    ----------
    model : SpectralModel
        A fitted specparam.SpectralModel (v2) object.
    signal : np.ndarray, shape (n_samples,)
        The time-domain signal to be reconstructed.
    sfreq : float
        Sampling frequency in Hz.
    component : {'full', 'aperiodic', 'periodic'}
        Which model component to use.
    out_of_range : {'passthrough', 'zero'}
        Policy for FFT bins outside the model's frequency range:
        'passthrough' (weight=1.0, signal passes through unchanged) or
        'zero' (weight=0.0, those frequencies are zeroed out).
    eps : float
        Floor for empirical PSD in weight computation. Default 1e-20.
    max_weight : float
        Upper clamp on weights. Default 100.0.

    Returns
    -------
    weights : np.ndarray, shape (n_freqs,)
        Per-bin amplitude weights aligned to rfft(signal).
    """
    weights, _ = _compute_weights_and_fft(
        model, signal, sfreq,
        component=component, out_of_range=out_of_range,
        eps=eps, max_weight=max_weight,
    )
    return weights


def specparam_reconstruct(
    signal: np.ndarray,
    sfreq: float,
    model,
    component: Component = "full",
    out_of_range: OutOfRangePolicy = "passthrough",
    eps: float = 1e-20,
    max_weight: float = 100.0,
) -> ReconstructionResult:
    """Reconstruct a time-domain signal component using a fitted SpecParam model.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Original time-domain signal.
    sfreq : float
        Sampling frequency in Hz.
    model : SpectralModel
        A fitted specparam.SpectralModel (v2) object.
    component : {'full', 'aperiodic', 'periodic'}
        Which model component to reconstruct.
    out_of_range : {'passthrough', 'zero'}
        Policy for bins outside model range: 'passthrough' or 'zero'.
    eps : float
        Floor for empirical PSD. Default 1e-20.
    max_weight : float
        Upper clamp on weights. Default 100.0.

    Returns
    -------
    result : ReconstructionResult
        Named tuple with .reconstruction and .residual arrays.
    """
    signal = np.asarray(signal, dtype=np.float64)
    weights, fft_coeffs = _compute_weights_and_fft(
        model, signal, sfreq,
        component=component, out_of_range=out_of_range,
        eps=eps, max_weight=max_weight,
    )
    reconstruction = np.fft.irfft(fft_coeffs * weights, n=len(signal))
    return ReconstructionResult(
        reconstruction=reconstruction,
        residual=signal - reconstruction,
    )
