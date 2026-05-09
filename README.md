# specparam-fft-weights

Time-domain signal reconstruction from [SpecParam](https://github.com/fooof-tools/fooof) spectral models via FFT weighting.

SpecParam (formerly FOOOF) fits parametric models to power spectra, decomposing them into aperiodic (1/f) and periodic (oscillatory) components. This package bridges those frequency-domain models back to the time domain: given a fitted `SpectralModel` and the original signal, it reconstructs the time-domain waveform corresponding to any model component.

## How it works

1. Fit a `SpectralModel` to the signal's power spectrum (via Welch's method)
2. Interpolate the model PSD onto the signal's FFT frequency grid
3. Compute amplitude-ratio weights: `weight[i] = sqrt(model[i] / empirical[i])`
4. Multiply the signal's complex FFT by these weights (phases untouched)
5. Inverse FFT back to the time domain

The result is a time-domain signal whose power spectrum matches the model's spectral shape, with the original signal's phase structure preserved.

## Installation

```bash
pip install numpy scipy specparam
```

Then copy `specparam_fft_weights.py` into your project, or install from source:

```bash
git clone https://github.com/chchatham/specparam-fft-weights.git
cd specparam-fft-weights
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from scipy.signal import welch
from specparam import SpectralModel
from specparam_fft_weights import specparam_reconstruct

# Load or generate your signal
sfreq = 256.0
signal = ...  # shape (n_samples,)

# Fit SpecParam to Welch PSD
freqs, psd = welch(signal, fs=sfreq, nperseg=512)
fm = SpectralModel(verbose=False)
fm.fit(freqs, psd, [1, 50])

# Extract aperiodic (1/f) reconstruction
result = specparam_reconstruct(signal, sfreq, fm, component='aperiodic')
aperiodic_signal = result.reconstruction
oscillatory_residual = result.residual  # contains rhythmic activity

# Extract periodic (oscillatory) reconstruction
result_p = specparam_reconstruct(signal, sfreq, fm, component='periodic')
periodic_signal = result_p.reconstruction
aperiodic_residual = result_p.residual  # contains 1/f background
```

## Documentation

Full tutorial and API reference: **[chchatham.github.io/specparam-fft-weights](https://chchatham.github.io/specparam-fft-weights/)**

## API overview

| Function | Description |
|---|---|
| `specparam_reconstruct(signal, sfreq, model, component)` | End-to-end: returns `(reconstruction, residual)` |
| `weights_from_specparam(model, signal, sfreq, component)` | Compute per-bin FFT amplitude weights from a fitted model |
| `model_psd_to_weights(model_psd, empirical_psd)` | Core weight computation from PSD arrays (no specparam dependency) |
| `apply_spectral_weights(signal, weights)` | Apply pre-computed weights to a signal's FFT |
| `extract_residual(signal, reconstruction)` | `signal - reconstruction` with validation |

## Citation

This package builds on SpecParam (formerly FOOOF). If you use this tool in published work, please cite the original paper:

> Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Walber JD, Knight RT, Shestyuk A, Voytek B (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655-1665. DOI: [10.1038/s41593-020-00744-x](https://doi.org/10.1038/s41593-020-00744-x)

## License

MIT
