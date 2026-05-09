---
layout: default
---

# specparam-fft-weights

**Time-domain signal reconstruction from SpecParam spectral models via FFT weighting.**

[SpecParam](https://github.com/fooof-tools/fooof) (formerly FOOOF) fits parametric models to power spectra, decomposing them into aperiodic (1/f) and periodic (oscillatory) components. `specparam-fft-weights` bridges those frequency-domain fits back to the time domain, enabling you to reconstruct the time-domain waveform corresponding to any model component while preserving the original signal's phase structure.

---

## Why this tool exists

SpecParam models the power spectrum, but many EEG analyses require time-domain signals: event-related analyses, connectivity measures, time-frequency decomposition, and trial-level statistics all operate on waveforms, not spectra. This tool lets you:

- **Isolate the aperiodic (1/f) time-domain signal** and subtract it to get a clean oscillatory residual
- **Extract the periodic component** to study oscillatory waveform morphology
- **Decompose a signal** into model-defined spectral components while preserving temporal structure (phase)

---

## Installation

```bash
pip install numpy scipy specparam
```

Clone and install from source:

```bash
git clone https://github.com/chchatham/specparam-fft-weights.git
cd specparam-fft-weights
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, NumPy, SciPy. SpecParam (`specparam>=2.0`) is required for the integration functions but not for the core weighting pipeline.

---

## Mathematical background

### The weight formula

Given a model PSD $P_{\text{model}}(f)$ and the signal's empirical periodogram $P_{\text{emp}}(f) = \lvert X(f) \rvert^2$, the per-bin amplitude weight is:

$$
w(f) = \sqrt{\frac{P_{\text{model}}(f)}{P_{\text{emp}}(f)}}
$$

### Spectral weighting

The signal's FFT coefficients $X(f)$ are multiplied by these real, non-negative weights:

$$
\hat{X}(f) = w(f) \cdot X(f)
$$

Since $w(f)$ is real and non-negative, **only the magnitude is scaled** — the complex phase $\angle X(f)$ is preserved exactly. The reconstruction is $\hat{x}(t) = \text{IFFT}[\hat{X}(f)]$.

### Residual

The residual $r(t) = x(t) - \hat{x}(t)$ contains everything the model did not capture. For an aperiodic reconstruction, the residual contains oscillatory content plus noise. For a periodic reconstruction, the residual contains the 1/f background.

### Normalization

SpecParam's model PSD comes from a Welch estimate (windowed, averaged), while the empirical periodogram $\lvert \text{rfft}(x) \rvert^2$ has different normalization. A global scale factor is computed from the ratio of total in-range power to align the two before computing weights.

---

## Tutorial: EEG aperiodic/periodic decomposition

This tutorial walks through a complete EEG workflow: generating a realistic synthetic signal, fitting SpecParam, and extracting time-domain aperiodic and periodic components.

### Step 1: Generate a synthetic EEG-like signal

```python
import numpy as np
from scipy.signal import welch

# Simulate 4 seconds of EEG at 256 Hz
sfreq = 256.0
duration = 4.0
n = int(sfreq * duration)
t = np.arange(n) / sfreq
rng = np.random.default_rng(42)

# 1/f^1.5 aperiodic background (colored noise)
white = rng.standard_normal(n)
freqs_fft = np.fft.rfftfreq(n, 1.0 / sfreq)
white_fft = np.fft.rfft(white)
scale = np.ones_like(freqs_fft)
scale[1:] = freqs_fft[1:] ** (-1.5 / 2)  # amplitude scaling for 1/f^1.5
pink = np.fft.irfft(white_fft * scale, n=n)
pink /= np.std(pink)

# Add a 10 Hz alpha oscillation
alpha = 2.5 * np.sin(2 * np.pi * 10 * t)

# Combined signal
signal = pink + alpha
```

### Step 2: Fit SpecParam to the power spectrum

```python
from specparam import SpectralModel

# Compute Welch PSD
freqs_welch, psd_welch = welch(signal, fs=sfreq, nperseg=512)

# Fit the spectral model
fm = SpectralModel(verbose=False)
fm.fit(freqs_welch, psd_welch, [1, 50])  # fit range: 1-50 Hz
```

### Step 3: Extract time-domain components

```python
from specparam_fft_weights import specparam_reconstruct

# Aperiodic reconstruction: the 1/f background in the time domain
result_ap = specparam_reconstruct(signal, sfreq, fm, component='aperiodic')
aperiodic_signal = result_ap.reconstruction
oscillatory_residual = result_ap.residual

# Periodic reconstruction: the oscillatory content in the time domain
result_pe = specparam_reconstruct(signal, sfreq, fm, component='periodic')
periodic_signal = result_pe.reconstruction
aperiodic_residual = result_pe.residual

# Full model reconstruction
result_full = specparam_reconstruct(signal, sfreq, fm, component='full')
```

### Step 4: Verify the decomposition

```python
# Reconstruction + residual always equals the original (sample-exact)
assert np.allclose(result_ap.reconstruction + result_ap.residual, signal, atol=1e-10)
assert np.allclose(result_pe.reconstruction + result_pe.residual, signal, atol=1e-10)
```

### Step 5: Visualize (optional)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
time_slice = slice(0, int(1.0 * sfreq))  # first 1 second

axes[0].plot(t[time_slice], signal[time_slice], 'k', alpha=0.7)
axes[0].set_title('Original signal')

axes[1].plot(t[time_slice], aperiodic_signal[time_slice], 'b')
axes[1].set_title('Aperiodic reconstruction (1/f background)')

axes[2].plot(t[time_slice], oscillatory_residual[time_slice], 'r')
axes[2].set_title('Oscillatory residual (alpha + noise)')

axes[3].plot(t[time_slice], periodic_signal[time_slice], 'g')
axes[3].set_title('Periodic reconstruction (oscillatory component)')

for ax in axes:
    ax.set_ylabel('Amplitude')
axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
```

---

## Advanced EEG workflows

### Workflow 1: Isolating oscillatory waveform morphology

Extract the periodic component to study waveform shape (e.g., alpha asymmetry) without 1/f contamination:

```python
result = specparam_reconstruct(signal, sfreq, fm, component='periodic')

# The periodic reconstruction preserves the original phase structure
# but reshapes magnitudes to match only the peak components
periodic_signal = result.reconstruction

# Bandpass the periodic signal for alpha-specific analysis
from scipy.signal import butter, sosfiltfilt

sos = butter(4, [8, 13], btype='band', fs=sfreq, output='sos')
alpha_waveform = sosfiltfilt(sos, periodic_signal)
```

### Workflow 2: 1/f-corrected event-related analysis

Remove the aperiodic background before computing ERPs or time-frequency representations:

```python
# Per-epoch decomposition
for epoch in epochs:  # shape: (n_epochs, n_samples)
    result = specparam_reconstruct(epoch, sfreq, fm, component='aperiodic')
    corrected_epoch = result.residual  # oscillatory content only
    # ... compute ERP, ERSP, etc. on corrected_epoch
```

### Workflow 3: Comparing aperiodic dynamics across conditions

Reconstruct the aperiodic component for each condition and compare:

```python
# Fit separate models per condition (or use one shared model)
fm_rest = SpectralModel(verbose=False)
fm_task = SpectralModel(verbose=False)

freqs_r, psd_r = welch(signal_rest, fs=sfreq, nperseg=512)
freqs_t, psd_t = welch(signal_task, fs=sfreq, nperseg=512)
fm_rest.fit(freqs_r, psd_r, [1, 50])
fm_task.fit(freqs_t, psd_t, [1, 50])

# Extract aperiodic components
ap_rest = specparam_reconstruct(signal_rest, sfreq, fm_rest, component='aperiodic')
ap_task = specparam_reconstruct(signal_task, sfreq, fm_task, component='aperiodic')

# Compare aperiodic time-domain properties
print(f"Rest aperiodic RMS: {np.sqrt(np.mean(ap_rest.reconstruction**2)):.4f}")
print(f"Task aperiodic RMS: {np.sqrt(np.mean(ap_task.reconstruction**2)):.4f}")
```

### Workflow 4: Handling out-of-range frequencies

By default, frequencies outside the SpecParam fit range pass through unchanged (`out_of_range='passthrough'`). To zero them out instead:

```python
# Zero out content below 1 Hz and above 50 Hz
result = specparam_reconstruct(
    signal, sfreq, fm,
    component='full',
    out_of_range='zero',
)
# result.reconstruction contains only the modeled frequency range
```

### Workflow 5: Lower-level control with weights

For custom pipelines, use `weights_from_specparam` to get the weight vector and apply it yourself:

```python
from specparam_fft_weights import weights_from_specparam, apply_spectral_weights

weights = weights_from_specparam(fm, signal, sfreq, component='aperiodic')

# Inspect or modify weights before applying
print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

# Apply weights manually
reconstruction = apply_spectral_weights(signal, weights)
```

---

## API reference

### `specparam_reconstruct(signal, sfreq, model, component='full', out_of_range='passthrough', eps=1e-20, max_weight=100.0)`

End-to-end reconstruction. Returns a `ReconstructionResult` named tuple with `.reconstruction` and `.residual` arrays.

**Parameters:**
- `signal` — 1-D numpy array, the time-domain signal
- `sfreq` — sampling frequency in Hz
- `model` — a fitted `specparam.SpectralModel` (v2) object
- `component` — `'full'`, `'aperiodic'`, or `'periodic'`
- `out_of_range` — `'passthrough'` (weight=1.0 outside fit range) or `'zero'` (weight=0.0)
- `eps` — floor for near-zero PSD values (default `1e-20`)
- `max_weight` — upper clamp on weights (default `100.0`)

### `weights_from_specparam(model, signal, sfreq, component='full', out_of_range='passthrough', eps=1e-20, max_weight=100.0)`

Compute per-bin FFT amplitude weights from a fitted SpecParam model. Returns a 1-D weight array aligned to `np.fft.rfft(signal)`.

### `model_psd_to_weights(model_psd, empirical_psd, eps=1e-20, max_weight=100.0)`

Core weight computation from PSD arrays (no specparam dependency). Both inputs must be in linear power (not log10), same shape. Returns `sqrt(model / empirical)`, clamped.

### `apply_spectral_weights(signal, weights)`

Apply pre-computed weights to a signal's FFT. Computes `rfft(signal)`, multiplies by `weights`, returns `irfft`. Phase is preserved exactly.

### `extract_residual(signal, reconstruction)`

Returns `signal - reconstruction` with shape validation.

### `ReconstructionResult`

Named tuple with fields:
- `.reconstruction` — the model-weighted time-domain signal
- `.residual` — `original - reconstruction`

---

## Limitations

- **Stationarity assumption.** The decomposition assumes the signal's spectral content is approximately stationary over the analysis window. For non-stationary signals, consider epoch-level decomposition.
- **Welch vs. periodogram.** The model PSD comes from a smoothed Welch estimate; the reconstruction operates on the raw periodogram. Weights reshape the spectrum toward the smooth model, which removes bin-to-bin periodogram noise.
- **Model accuracy.** Decomposition quality is bounded by SpecParam's fit quality. Always check `fm.results.metrics` (r-squared, error) before trusting the decomposition.
- **Out-of-range frequencies.** Bins outside the SpecParam fit range are handled by policy (`passthrough` or `zero`), not by the model. Neither is a true extrapolation.
- **SpecParam v2 only.** This package requires `specparam>=2.0` (the v2 API). The v1 `fooof` package uses a different attribute structure and is not supported.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

71 tests covering core weighting, SpecParam integration, numerical integrity (phase preservation, Parseval energy accounting, DC/Nyquist handling), and synthetic-signal integration.

---

## Citation

This package builds on SpecParam (formerly FOOOF). If you use this tool in published work, please cite the original paper:

> Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Walber JD, Knight RT, Shestyuk A, Voytek B (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655-1665. DOI: [10.1038/s41593-020-00744-x](https://doi.org/10.1038/s41593-020-00744-x)

---

[View on GitHub](https://github.com/chchatham/specparam-fft-weights)
