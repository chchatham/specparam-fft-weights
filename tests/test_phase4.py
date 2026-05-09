"""Integration tests with synthetic signals.

Generate known 1/f + alpha peak signals, fit SpecParam, extract components,
and verify the decomposition makes physical sense.
"""

import numpy as np
import pytest

from specparam_fft_weights import specparam_reconstruct

from .conftest import fit_specparam_to_signal, make_synthetic_signal


class TestAperiodicExtraction:

    def test_residual_shows_alpha_peak(self, synthetic_setup):
        signal, sfreq, fm = synthetic_setup
        result = specparam_reconstruct(signal, sfreq, fm, component="aperiodic")

        freqs_fft = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        residual_fft = np.fft.rfft(result.residual)
        residual_psd = residual_fft.real ** 2 + residual_fft.imag ** 2

        alpha_band = (freqs_fft >= 8.0) & (freqs_fft <= 12.0)
        non_alpha = (freqs_fft >= 20.0) & (freqs_fft <= 40.0)

        alpha_power = np.mean(residual_psd[alpha_band])
        background_power = np.mean(residual_psd[non_alpha])

        assert alpha_power > 5 * background_power, (
            f"Alpha power {alpha_power:.2f} should be >> background {background_power:.2f}"
        )

    def test_residual_contains_more_alpha_than_reconstruction(self, synthetic_setup):
        signal, sfreq, fm = synthetic_setup
        result = specparam_reconstruct(signal, sfreq, fm, component="aperiodic")

        freqs_fft = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        recon_psd = np.abs(np.fft.rfft(result.reconstruction)) ** 2
        residual_psd = np.abs(np.fft.rfft(result.residual)) ** 2

        alpha_band = (freqs_fft >= 9.0) & (freqs_fft <= 11.0)
        nearby_band = (freqs_fft >= 20.0) & (freqs_fft <= 30.0)

        recon_ratio = np.mean(recon_psd[alpha_band]) / np.mean(recon_psd[nearby_band])
        resid_ratio = np.mean(residual_psd[alpha_band]) / np.mean(residual_psd[nearby_band])

        assert resid_ratio > recon_ratio, (
            f"Residual should have stronger alpha peak: "
            f"residual ratio {resid_ratio:.1f} vs recon ratio {recon_ratio:.1f}"
        )


class TestPeriodicExtraction:

    def test_residual_shows_1f(self, synthetic_setup):
        signal, sfreq, fm = synthetic_setup
        result = specparam_reconstruct(signal, sfreq, fm, component="periodic")

        freqs_fft = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        residual_psd = np.abs(np.fft.rfft(result.residual)) ** 2

        low_band = (freqs_fft >= 2.0) & (freqs_fft <= 5.0)
        high_band = (freqs_fft >= 30.0) & (freqs_fft <= 60.0)

        assert np.mean(residual_psd[low_band]) > np.mean(residual_psd[high_band])

    def test_residual_has_reduced_alpha(self, synthetic_setup):
        signal, sfreq, fm = synthetic_setup
        result = specparam_reconstruct(signal, sfreq, fm, component="periodic")

        freqs_fft = np.fft.rfftfreq(len(signal), 1.0 / sfreq)
        signal_psd = np.abs(np.fft.rfft(signal)) ** 2
        residual_psd = np.abs(np.fft.rfft(result.residual)) ** 2

        alpha_band = (freqs_fft >= 9.0) & (freqs_fft <= 11.0)
        assert np.mean(residual_psd[alpha_band]) < np.mean(signal_psd[alpha_band])


class TestFullModelExtraction:

    def test_full_reconstruction_captures_signal_energy(self, synthetic_setup):
        signal, sfreq, fm = synthetic_setup
        result = specparam_reconstruct(signal, sfreq, fm, component="full")

        signal_energy = np.sum(signal ** 2)
        recon_energy = np.sum(result.reconstruction ** 2)
        residual_energy = np.sum(result.residual ** 2)

        assert recon_energy > 0.3 * signal_energy
        assert residual_energy < signal_energy


class TestReconstructionPlusResidual:

    def test_sample_exact_sum(self, synthetic_setup):
        signal, sfreq, fm = synthetic_setup
        for comp in ["full", "aperiodic", "periodic"]:
            result = specparam_reconstruct(signal, sfreq, fm, component=comp)
            np.testing.assert_allclose(
                result.reconstruction + result.residual, signal, atol=1e-10,
                err_msg=f"Sample-exact sum failed for component='{comp}'",
            )

    def test_with_different_signal_lengths(self):
        signal_long, sfreq = make_synthetic_signal(duration=10.0, seed=99)
        fm = fit_specparam_to_signal(signal_long, sfreq)

        for n in [1024, 2048, 4096, 3000, 3001]:
            sig = signal_long[:n]
            result = specparam_reconstruct(sig, sfreq, fm, component="full")
            np.testing.assert_allclose(
                result.reconstruction + result.residual, sig, atol=1e-10,
                err_msg=f"Failed for signal length {n}",
            )
