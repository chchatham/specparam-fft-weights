"""Shared fixtures and helpers for specparam_fft_weights tests."""

import numpy as np
import pytest
from scipy.signal import welch

from specparam import SpectralModel


def make_synthetic_signal(
    sfreq=512.0,
    duration=8.0,
    exponent=1.5,
    alpha_freq=10.0,
    alpha_amplitude=3.0,
    seed=42,
):
    """Generate 1/f noise + alpha oscillation.

    Returns (signal, sfreq).
    """
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    t = np.arange(n) / sfreq

    white = rng.standard_normal(n)
    freqs_fft = np.fft.rfftfreq(n, 1.0 / sfreq)
    white_fft = np.fft.rfft(white)
    scale = np.ones_like(freqs_fft)
    scale[1:] = freqs_fft[1:] ** (-exponent / 2)
    pink = np.fft.irfft(white_fft * scale, n=n)
    pink /= np.std(pink)

    alpha = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
    return pink + alpha, sfreq


def fit_specparam_to_signal(signal, sfreq, freq_range=(1, 80)):
    """Compute Welch PSD and fit SpecParam."""
    nperseg = min(1024, len(signal) // 2)
    freqs_w, psd_w = welch(signal, fs=sfreq, nperseg=nperseg)
    fm = SpectralModel(verbose=False)
    fm.fit(freqs_w, psd_w, list(freq_range))
    return fm


@pytest.fixture
def fitted_model():
    """Fitted SpectralModel on synthetic 1/f + alpha signal (short, for fast tests)."""
    signal, sfreq = make_synthetic_signal(
        sfreq=256.0, duration=4.0, alpha_amplitude=2.0, seed=42,
    )
    fm = fit_specparam_to_signal(signal, sfreq, freq_range=(1, 50))
    return fm, signal, sfreq


@pytest.fixture
def synthetic_setup():
    """Synthetic 1/f + alpha signal with fitted model (longer, for integration tests)."""
    signal, sfreq = make_synthetic_signal()
    fm = fit_specparam_to_signal(signal, sfreq)
    return signal, sfreq, fm
