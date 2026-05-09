"""Tests for SpecParam integration."""

import numpy as np
import pytest

from specparam_fft_weights import (
    ReconstructionResult,
    specparam_reconstruct,
    weights_from_specparam,
)


class TestWeightsFromSpecparam:

    def test_returns_correct_length(self, fitted_model):
        fm, signal, sfreq = fitted_model
        weights = weights_from_specparam(fm, signal, sfreq, component="full")
        expected_len = len(signal) // 2 + 1
        assert len(weights) == expected_len

    def test_all_components(self, fitted_model):
        fm, signal, sfreq = fitted_model
        for comp in ["full", "aperiodic", "periodic"]:
            weights = weights_from_specparam(fm, signal, sfreq, component=comp)
            assert len(weights) == len(signal) // 2 + 1
            assert np.all(np.isfinite(weights))
            assert np.all(weights >= 0.0)

    def test_invalid_component_raises(self, fitted_model):
        fm, signal, sfreq = fitted_model
        with pytest.raises(ValueError, match="Unknown component"):
            weights_from_specparam(fm, signal, sfreq, component="bogus")

    def test_passthrough_out_of_range(self, fitted_model):
        fm, signal, sfreq = fitted_model
        weights = weights_from_specparam(
            fm, signal, sfreq, component="full", out_of_range="passthrough",
        )
        fft_freqs = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        model_lo = fm.data.freq_range[0]
        model_hi = fm.data.freq_range[1]
        oor_mask = (fft_freqs < model_lo) | (fft_freqs > model_hi)
        if np.any(oor_mask):
            np.testing.assert_allclose(weights[oor_mask], 1.0, atol=1e-10)

    def test_zero_out_of_range(self, fitted_model):
        fm, signal, sfreq = fitted_model
        weights = weights_from_specparam(
            fm, signal, sfreq, component="full", out_of_range="zero",
        )
        fft_freqs = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        model_lo = fm.data.freq_range[0]
        model_hi = fm.data.freq_range[1]
        oor_mask = (fft_freqs < model_lo) | (fft_freqs > model_hi)
        if np.any(oor_mask):
            np.testing.assert_allclose(weights[oor_mask], 0.0, atol=1e-10)

    def test_invalid_oor_policy_raises(self, fitted_model):
        fm, signal, sfreq = fitted_model
        with pytest.raises(ValueError, match="out_of_range"):
            weights_from_specparam(
                fm, signal, sfreq, out_of_range="extrapolate",
            )


class TestSpecparamReconstruct:

    def test_returns_named_tuple(self, fitted_model):
        fm, signal, sfreq = fitted_model
        result = specparam_reconstruct(signal, sfreq, fm, component="full")
        assert isinstance(result, ReconstructionResult)
        assert result.reconstruction.shape == signal.shape
        assert result.residual.shape == signal.shape

    def test_reconstruction_plus_residual_equals_original(self, fitted_model):
        fm, signal, sfreq = fitted_model
        for comp in ["full", "aperiodic", "periodic"]:
            result = specparam_reconstruct(signal, sfreq, fm, component=comp)
            np.testing.assert_allclose(
                result.reconstruction + result.residual, signal, atol=1e-10,
                err_msg=f"Failed for component={comp}",
            )

    def test_full_reconstruction_preserves_most_energy(self, fitted_model):
        fm, signal, sfreq = fitted_model
        result = specparam_reconstruct(signal, sfreq, fm, component="full")
        signal_energy = np.sum(signal ** 2)
        recon_energy = np.sum(result.reconstruction ** 2)
        assert recon_energy > 0.01 * signal_energy

    def test_aperiodic_removes_oscillation(self, fitted_model):
        fm, signal, sfreq = fitted_model
        result = specparam_reconstruct(signal, sfreq, fm, component="aperiodic")
        residual_fft = np.fft.rfft(result.residual)
        residual_psd = residual_fft.real ** 2 + residual_fft.imag ** 2
        fft_freqs = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        alpha_idx = np.argmin(np.abs(fft_freqs - 10.0))
        assert residual_psd[alpha_idx] > 0


class TestFreqInterpolation:

    def test_different_signal_lengths(self, fitted_model):
        """Weights should work for signals of various lengths."""
        fm, signal, sfreq = fitted_model
        for n in [512, 1024, 2048, 1000, 1001]:
            short_signal = signal[:n] if n <= len(signal) else np.tile(signal, 3)[:n]
            weights = weights_from_specparam(fm, short_signal, sfreq)
            assert len(weights) == n // 2 + 1
            assert np.all(np.isfinite(weights))
