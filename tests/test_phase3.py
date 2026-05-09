"""Tests for Phase 3: numerical integrity."""

import numpy as np
import pytest

from specparam_fft_weights import (
    apply_spectral_weights,
    extract_residual,
    model_psd_to_weights,
)


class TestPhasePreservation:

    @pytest.mark.parametrize("n", [128, 255, 256, 1000, 1023, 1024, 2048])
    def test_phase_exact_match(self, n):
        """angle(weighted_fft) == angle(original_fft) for all non-zero bins."""
        rng = np.random.default_rng(n)
        signal = rng.standard_normal(n)
        n_freqs = n // 2 + 1
        weights = rng.uniform(0.1, 5.0, size=n_freqs)

        original_fft = np.fft.rfft(signal)
        weighted_fft = original_fft * weights

        nonzero = np.abs(original_fft) > 1e-15
        np.testing.assert_allclose(
            np.angle(weighted_fft[nonzero]),
            np.angle(original_fft[nonzero]),
            atol=1e-12,
        )

    def test_phase_preserved_through_full_pipeline(self):
        rng = np.random.default_rng(77)
        signal = rng.standard_normal(512)
        n_freqs = len(signal) // 2 + 1
        fft_coeffs = np.fft.rfft(signal)
        empirical_psd = np.abs(fft_coeffs) ** 2
        model_psd = empirical_psd * rng.uniform(0.5, 2.0, size=n_freqs)

        weights = model_psd_to_weights(model_psd, empirical_psd)
        weighted_fft = fft_coeffs * weights

        nonzero = np.abs(fft_coeffs) > 1e-15
        np.testing.assert_allclose(
            np.angle(weighted_fft[nonzero]),
            np.angle(fft_coeffs[nonzero]),
            atol=1e-12,
        )


class TestEnergyAccounting:

    @pytest.mark.parametrize("n", [256, 512, 1000, 1023])
    def test_parseval_weighted_signal(self, n):
        """Verify Parseval: time-domain energy == freq-domain energy."""
        rng = np.random.default_rng(n + 100)
        signal = rng.standard_normal(n)
        n_freqs = n // 2 + 1
        weights = rng.uniform(0.2, 3.0, size=n_freqs)

        reconstruction = apply_spectral_weights(signal, weights)

        time_energy = np.sum(reconstruction ** 2)

        recon_fft = np.fft.rfft(reconstruction)
        recon_psd = np.abs(recon_fft) ** 2
        # Parseval for rfft: sum includes DC once, Nyquist once (if even),
        # and all other bins doubled
        freq_energy = recon_psd[0]
        if n % 2 == 0:
            freq_energy += recon_psd[-1]
            freq_energy += 2 * np.sum(recon_psd[1:-1])
        else:
            freq_energy += 2 * np.sum(recon_psd[1:])
        freq_energy /= n

        np.testing.assert_allclose(time_energy, freq_energy, rtol=1e-10)

    def test_energy_decomposition(self):
        """recon + residual == original, so energies relate via cross-term."""
        rng = np.random.default_rng(999)
        signal = rng.standard_normal(1024)
        n_freqs = len(signal) // 2 + 1

        fft_coeffs = np.fft.rfft(signal)
        empirical_psd = np.abs(fft_coeffs) ** 2
        model_psd = empirical_psd * rng.uniform(0.3, 1.5, size=n_freqs)

        weights = model_psd_to_weights(model_psd, empirical_psd)
        reconstruction = apply_spectral_weights(signal, weights)
        residual = extract_residual(signal, reconstruction)

        # signal = recon + residual (exact)
        np.testing.assert_allclose(reconstruction + residual, signal, atol=1e-12)

        # |signal|^2 = |recon|^2 + |residual|^2 + 2*recon.residual
        e_signal = np.sum(signal ** 2)
        e_recon = np.sum(reconstruction ** 2)
        e_resid = np.sum(residual ** 2)
        cross = 2 * np.sum(reconstruction * residual)
        np.testing.assert_allclose(e_signal, e_recon + e_resid + cross, rtol=1e-10)


class TestEvenOddLength:

    @pytest.mark.parametrize("n", [100, 101, 256, 257, 1024, 1025])
    def test_round_trip_identity(self, n):
        rng = np.random.default_rng(n)
        signal = rng.standard_normal(n)
        weights = np.ones(n // 2 + 1)
        result = apply_spectral_weights(signal, weights)
        np.testing.assert_allclose(result, signal, atol=1e-12)

    @pytest.mark.parametrize("n", [100, 101, 256, 257])
    def test_output_length_matches_input(self, n):
        rng = np.random.default_rng(n)
        signal = rng.standard_normal(n)
        weights = rng.uniform(0.5, 2.0, size=n // 2 + 1)
        result = apply_spectral_weights(signal, weights)
        assert len(result) == n

    def test_nyquist_bin_exists_only_for_even(self):
        # Even: n=8 → rfft has 5 bins (0, 1, 2, 3, 4) where bin 4 is Nyquist
        signal_even = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
        fft_even = np.fft.rfft(signal_even)
        assert len(fft_even) == 5
        assert fft_even[-1].imag == 0.0

        # Odd: n=7 → rfft has 4 bins (0, 1, 2, 3), no Nyquist
        signal_odd = np.array([1.0, 2, 3, 4, 5, 6, 7])
        fft_odd = np.fft.rfft(signal_odd)
        assert len(fft_odd) == 4


class TestDCBinHandling:

    def test_dc_bin_stays_real_after_weighting(self):
        rng = np.random.default_rng(42)
        for n in [64, 128, 255, 256, 1024]:
            signal = rng.standard_normal(n)
            fft_coeffs = np.fft.rfft(signal)
            weights = rng.uniform(0.1, 10.0, size=n // 2 + 1)
            weighted = fft_coeffs * weights
            assert weighted[0].imag == 0.0, f"DC bin has nonzero imag for n={n}"

    def test_dc_bin_weight_scales_mean(self):
        signal = np.array([3.0, 3.0, 3.0, 3.0])
        weights = np.array([2.0, 1.0, 1.0])
        result = apply_spectral_weights(signal, weights)
        # DC component = mean * N = 12.0. weight=2.0 makes it 24.0. mean=6.0
        np.testing.assert_allclose(result.mean(), 6.0, atol=1e-14)


class TestNegativeNanInfPrevention:

    def test_nan_model_psd_rejected(self):
        model = np.array([1.0, np.nan, 2.0])
        empirical = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="Non-finite"):
            model_psd_to_weights(model, empirical)

    def test_inf_model_psd_clamped(self):
        model = np.array([np.inf])
        empirical = np.array([1.0])
        # sqrt(inf / 1) = inf, but clamped to max_weight
        weights = model_psd_to_weights(model, empirical, max_weight=50.0)
        assert weights[0] == 50.0

    def test_negative_empirical_handled(self):
        model = np.array([1.0])
        empirical = np.array([-1.0])
        # Negative empirical floored to eps
        weights = model_psd_to_weights(model, empirical)
        assert np.isfinite(weights[0])
        assert weights[0] >= 0.0

    def test_all_zero_empirical(self):
        model = np.array([1.0, 2.0, 3.0])
        empirical = np.zeros(3)
        weights = model_psd_to_weights(model, empirical)
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 100.0)

    def test_all_zero_model(self):
        model = np.zeros(5)
        empirical = np.ones(5)
        weights = model_psd_to_weights(model, empirical)
        np.testing.assert_allclose(weights, 0.0)

    def test_very_small_values(self):
        model = np.array([1e-300, 1e-300])
        empirical = np.array([1e-300, 1e-300])
        weights = model_psd_to_weights(model, empirical)
        assert np.all(np.isfinite(weights))
