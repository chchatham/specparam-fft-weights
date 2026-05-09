"""Tests for Phase 1: core weighting pipeline (pure numpy, no specparam)."""

import numpy as np
import pytest

from specparam_fft_weights import (
    apply_spectral_weights,
    extract_residual,
    model_psd_to_weights,
)


class TestModelPsdToWeights:

    def test_identity_weights_when_model_equals_empirical(self):
        psd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = model_psd_to_weights(psd, psd)
        np.testing.assert_allclose(weights, 1.0, atol=1e-15)

    def test_known_ratio(self):
        model = np.array([4.0, 9.0, 16.0])
        empirical = np.array([1.0, 1.0, 1.0])
        weights = model_psd_to_weights(model, empirical)
        np.testing.assert_allclose(weights, [2.0, 3.0, 4.0])

    def test_zero_empirical_uses_floor(self):
        model = np.array([1.0, 1.0])
        empirical = np.array([0.0, 1.0])
        weights = model_psd_to_weights(model, empirical)
        assert np.all(np.isfinite(weights))
        assert weights[0] <= 100.0

    def test_negative_model_floored_to_zero(self):
        model = np.array([-1.0, 4.0])
        empirical = np.array([1.0, 1.0])
        weights = model_psd_to_weights(model, empirical)
        assert weights[0] == 0.0
        assert weights[1] == 2.0

    def test_max_weight_clamp(self):
        model = np.array([1e30])
        empirical = np.array([1e-30])
        weights = model_psd_to_weights(model, empirical, max_weight=50.0)
        assert weights[0] == 50.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            model_psd_to_weights(np.ones(3), np.ones(4))

    def test_output_dtype_float64(self):
        weights = model_psd_to_weights(np.ones(5), np.ones(5))
        assert weights.dtype == np.float64

    def test_all_weights_non_negative(self):
        rng = np.random.default_rng(42)
        model = rng.exponential(size=100)
        empirical = rng.exponential(size=100)
        weights = model_psd_to_weights(model, empirical)
        assert np.all(weights >= 0.0)


class TestApplySpectralWeights:

    def test_round_trip_identity_even(self):
        """weights = 1.0 → output == input for even-length signal."""
        rng = np.random.default_rng(123)
        signal = rng.standard_normal(1024)
        n_freqs = len(signal) // 2 + 1
        weights = np.ones(n_freqs)
        result = apply_spectral_weights(signal, weights)
        np.testing.assert_allclose(result, signal, atol=1e-12)

    def test_round_trip_identity_odd(self):
        """weights = 1.0 → output == input for odd-length signal."""
        rng = np.random.default_rng(456)
        signal = rng.standard_normal(1023)
        n_freqs = len(signal) // 2 + 1
        weights = np.ones(n_freqs)
        result = apply_spectral_weights(signal, weights)
        np.testing.assert_allclose(result, signal, atol=1e-12)

    def test_phase_preservation(self):
        """Phases must be identical before and after weighting."""
        rng = np.random.default_rng(789)
        signal = rng.standard_normal(512)
        n_freqs = len(signal) // 2 + 1
        weights = rng.uniform(0.5, 2.0, size=n_freqs)

        original_fft = np.fft.rfft(signal)
        original_phase = np.angle(original_fft)

        weighted_fft = original_fft * weights
        weighted_phase = np.angle(weighted_fft)

        nonzero = np.abs(original_fft) > 1e-15
        np.testing.assert_allclose(
            weighted_phase[nonzero], original_phase[nonzero], atol=1e-12
        )

    def test_dc_bin_stays_real(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([2.0, 1.0, 0.5])
        result_fft = np.fft.rfft(signal) * weights
        assert result_fft[0].imag == 0.0

    def test_nyquist_bin_stays_real_even_length(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 2.0])
        result_fft = np.fft.rfft(signal) * weights
        assert result_fft[-1].imag == 0.0

    def test_zero_weights_zero_signal(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        n_freqs = len(signal) // 2 + 1
        weights = np.zeros(n_freqs)
        result = apply_spectral_weights(signal, weights)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_wrong_weight_length_raises(self):
        with pytest.raises(ValueError, match="weights length"):
            apply_spectral_weights(np.ones(10), np.ones(3))



class TestExtractResidual:

    def test_residual_is_difference(self):
        signal = np.array([1.0, 2.0, 3.0])
        recon = np.array([0.5, 1.5, 2.5])
        residual = extract_residual(signal, recon)
        np.testing.assert_allclose(residual, [0.5, 0.5, 0.5])

    def test_zero_residual_for_identical(self):
        signal = np.array([1.0, 2.0, 3.0])
        residual = extract_residual(signal, signal.copy())
        np.testing.assert_allclose(residual, 0.0, atol=1e-15)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            extract_residual(np.ones(3), np.ones(4))


class TestEndToEndPhase1:

    def test_full_pipeline_identity(self):
        """model == empirical → reconstruction == original, residual == 0."""
        rng = np.random.default_rng(999)
        signal = rng.standard_normal(2048)
        n_freqs = len(signal) // 2 + 1
        fft_coeffs = np.fft.rfft(signal)
        empirical_psd = np.abs(fft_coeffs) ** 2

        weights = model_psd_to_weights(empirical_psd, empirical_psd)
        np.testing.assert_allclose(weights, 1.0, atol=1e-15)

        reconstruction = apply_spectral_weights(signal, weights)
        np.testing.assert_allclose(reconstruction, signal, atol=1e-12)

        residual = extract_residual(signal, reconstruction)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_reconstruction_plus_residual_equals_original(self):
        """For arbitrary weights, recon + residual == original."""
        rng = np.random.default_rng(1234)
        signal = rng.standard_normal(500)
        n_freqs = len(signal) // 2 + 1

        fft_coeffs = np.fft.rfft(signal)
        empirical_psd = np.abs(fft_coeffs) ** 2
        model_psd = empirical_psd * rng.uniform(0.2, 1.8, size=n_freqs)

        weights = model_psd_to_weights(model_psd, empirical_psd)
        reconstruction = apply_spectral_weights(signal, weights)
        residual = extract_residual(signal, reconstruction)

        np.testing.assert_allclose(reconstruction + residual, signal, atol=1e-12)

    def test_energy_scaling(self):
        """Weighted signal energy should match model PSD total (Parseval)."""
        rng = np.random.default_rng(5678)
        n = 1024
        signal = rng.standard_normal(n)
        fft_coeffs = np.fft.rfft(signal)
        empirical_psd = np.abs(fft_coeffs) ** 2
        scale = 0.5
        model_psd = empirical_psd * scale

        weights = model_psd_to_weights(model_psd, empirical_psd)
        reconstruction = apply_spectral_weights(signal, weights)

        recon_fft = np.fft.rfft(reconstruction)
        recon_psd = np.abs(recon_fft) ** 2
        np.testing.assert_allclose(recon_psd, model_psd, rtol=1e-10)
