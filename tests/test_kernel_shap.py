"""Synthetic KernelSHAP checks: no datasets, checkpoints or saved maps."""

from math import factorial

import numpy as np
import pytest

from attribution.game_theoretic.kernel_shap import (
    KernelShapConfig,
    fit_kernel_shap,
    sample_coalitions,
    shapley_kernel_weight,
)


def test_kernel_formula_and_exact_enumeration():
    assert shapley_kernel_weight(4, 2) == pytest.approx(1 / 8)
    assert np.isinf(shapley_kernel_weight(4, 0))
    sample = sample_coalitions(3, KernelShapConfig(num_samples=8))
    assert sample.sampling_mode == "exact"
    assert sample.masks.shape == (8, 3)
    assert not sample.masks[0].any()
    assert sample.masks[1].all()
    assert np.unique(sample.masks, axis=0).shape[0] == 8
    assert np.all(sample.regression_weights[2:] > 0)


def test_exact_nonlinear_game_matches_brute_force_shapley():
    image = np.ones((1, 3, 1))
    segments = np.array([[11, 22, 33]])

    def value(z):
        return 5 + 2 * z[:, 0] - 3 * z[:, 1] + 4 * z[:, 2] + 6 * z[:, 0] * z[:, 1]

    def predict(batch):
        return value(batch[:, 0, :, 0])[:, None]

    result = fit_kernel_shap(
        image, segments, predict, 0, KernelShapConfig(num_samples=8, batch_size=3)
    )
    expected = np.zeros(3)
    for feature in range(3):
        others = [index for index in range(3) if index != feature]
        for encoded in range(1 << len(others)):
            coalition = np.zeros(3)
            chosen = [others[j] for j in range(len(others)) if encoded >> j & 1]
            coalition[chosen] = 1
            weight = factorial(len(chosen)) * factorial(2 - len(chosen)) / factorial(3)
            without = value(coalition[None])[0]
            coalition[feature] = 1
            expected[feature] += weight * (value(coalition[None])[0] - without)
    np.testing.assert_allclose(result.coefficients, expected, atol=1e-10)
    np.testing.assert_allclose(result.coefficients, [5, 0, 4], atol=1e-10)
    assert result.base_value == pytest.approx(5)
    assert result.original_logit == pytest.approx(14)
    assert result.local_prediction == pytest.approx(14)
    assert result.efficiency_residual == pytest.approx(0, abs=1e-14)
    assert result.forward_samples == 8
    assert result.forward_batches == 3


def test_approximate_additive_model_recovers_effects_and_budget():
    image = np.array([[[2.0], [-3.0]], [[1.0], [4.0]]])
    segments = np.array([[10, 20], [10, 30]])
    seen = []

    def predict(batch):
        seen.append(batch.copy())
        y = 7 + batch.sum(axis=(1, 2, 3))
        return np.column_stack([-y, y])

    # Six regions forces finite sampling at this budget; three are zero-area in
    # the image only conceptually, so use a six-pixel input for the actual test.
    image = np.arange(1, 7, dtype=float)[None, :, None]
    segments = np.array([[10, 20, 30, 40, 50, 60]])
    result = fit_kernel_shap(
        image,
        segments,
        predict,
        1,
        KernelShapConfig(num_samples=37, batch_size=8, seed=4),
    )
    np.testing.assert_allclose(result.coefficients, np.arange(1, 7), atol=1e-10)
    assert result.base_value == pytest.approx(7)
    assert result.original_logit == pytest.approx(28)
    assert result.forward_samples == sum(map(len, seen)) == 37
    assert result.forward_batches == len(seen) == 5
    assert result.sampling_mode == "paired_kernel"
    assert result.design_rank == 6
    assert np.isfinite(result.condition_number)
    assert result.weighted_r2 == pytest.approx(1)


def test_paired_sampling_is_reproducible_local_and_complementary():
    np.random.seed(19)
    state = np.random.get_state()
    config = KernelShapConfig(num_samples=31, seed=3)
    sample = sample_coalitions(8, config)
    np.testing.assert_array_equal(sample.masks, sample_coalitions(8, config).masks)
    assert not np.array_equal(
        sample.masks,
        sample_coalitions(8, KernelShapConfig(num_samples=31, seed=4)).masks,
    )
    for row in range(2, 30, 2):
        np.testing.assert_array_equal(sample.masks[row + 1], ~sample.masks[row])
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]


def test_batch_size_invariance_and_one_region_case():
    image = np.arange(1, 9, dtype=float)[None, :, None]
    segments = np.arange(8)[None]

    def predict(batch):
        return np.sin(batch.sum(axis=(1, 2, 3)))[:, None]

    a = fit_kernel_shap(
        image,
        segments,
        predict,
        0,
        KernelShapConfig(num_samples=41, batch_size=1, seed=2),
    )
    b = fit_kernel_shap(
        image,
        segments,
        predict,
        0,
        KernelShapConfig(num_samples=41, batch_size=9, seed=2),
    )
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.base_value == b.base_value
    assert a.original_logit == b.original_logit

    one = fit_kernel_shap(
        np.ones((2, 2, 1)),
        np.full((2, 2), -5),
        lambda batch: (3 + batch.sum(axis=(1, 2, 3)))[:, None],
        0,
    )
    np.testing.assert_allclose(one.coefficients, [4])
    assert one.forward_samples == 2
    assert one.sampling_mode == "exact"


def test_low_budget_reports_rank_deficiency_but_preserves_efficiency():
    image = np.ones((1, 10, 1))
    segments = np.arange(10)[None]
    result = fit_kernel_shap(
        image,
        segments,
        lambda batch: batch.sum(axis=(1, 2, 3))[:, None],
        0,
        KernelShapConfig(num_samples=3),
    )
    assert result.design_rank < 10
    assert np.isinf(result.condition_number)
    assert result.coefficients.sum() == pytest.approx(10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 1},
        {"batch_size": 0},
        {"seed": -1},
        {"seed": True},
        {"num_samples": 3.5},
        {"ridge_alpha": -1},
        {"ridge_alpha": float("nan")},
        {"paired_sampling": 1},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        KernelShapConfig(**kwargs)


@pytest.mark.parametrize(
    "image, segments, target",
    [
        (np.ones((2, 2)), np.zeros((2, 2), dtype=int), 0),
        (np.ones((2, 2, 1), dtype=int), np.zeros((2, 2), dtype=int), 0),
        (np.ones((2, 2, 1)), np.zeros((1, 2), dtype=int), 0),
        (np.ones((2, 2, 1)), np.zeros((2, 2)), 0),
        (np.ones((2, 2, 1)), np.zeros((2, 2), dtype=int), -1),
    ],
)
def test_invalid_inputs_fail_before_model_call(image, segments, target):
    def predict(batch):
        pytest.fail("invalid input must not consume forwards")

    with pytest.raises(ValueError):
        fit_kernel_shap(image, segments, predict, target)


@pytest.mark.parametrize(
    "output",
    [np.zeros(3), np.zeros((3, 0)), np.full((3, 1), np.nan), np.zeros((1, 1))],
)
def test_bad_prediction_contract(output):
    with pytest.raises(ValueError, match="predict_logits"):
        fit_kernel_shap(
            np.ones((1, 3, 1)),
            np.arange(3)[None],
            lambda batch: output,
            0,
            KernelShapConfig(num_samples=3),
        )
