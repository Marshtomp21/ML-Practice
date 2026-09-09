"""Synthetic numerical checks: no checkpoints, dataset access or saved maps."""
from itertools import product

import numpy as np
import pytest

from attribution.perturbation.lime import LimeConfig, fit_lime, locality_weights, sample_masks
from preprocessing.superpixels import slic_segments


def test_additive_model_recovers_signed_region_effects_and_budget():
    image = np.array([[[2.0], [-3.0]], [[1.0], [4.0]]])
    segments = np.array([[10, 20], [10, 30]])
    seen = []

    def predict(batch):
        seen.append(batch.copy())
        y = 7 + batch.sum(axis=(1, 2, 3))
        return np.column_stack([-y, y])

    original = image.copy()
    result = fit_lime(image, segments, predict, 1,
                      LimeConfig(num_samples=257, batch_size=31, ridge_alpha=0))
    np.testing.assert_allclose(result.coefficients, [3, -3, 4], atol=1e-10)
    assert result.intercept == pytest.approx(7)
    assert result.original_logit == pytest.approx(11)
    assert result.local_prediction == pytest.approx(11)
    assert result.weighted_r2 == pytest.approx(1)
    assert result.forward_samples == sum(len(b) for b in seen) == 257
    assert result.forward_batches == len(seen) == 9
    np.testing.assert_array_equal(seen[0][0], original)
    np.testing.assert_array_equal(image, original)
    perturbed = np.concatenate(seen)
    assert np.all((perturbed == 0) | (perturbed == image))
    np.testing.assert_array_equal(perturbed[:, 0, 0, 0] != 0, perturbed[:, 1, 0, 0] != 0)


def test_kernel_matches_cosine_reference_including_empty_mask():
    from sklearn.metrics import pairwise_distances
    masks = np.array(list(product([0, 1], repeat=3)))
    distances = pairwise_distances(masks, np.ones((1, 3)), metric="cosine").ravel()
    np.testing.assert_allclose(locality_weights(masks, 0.25),
                               np.sqrt(np.exp(-distances**2 / 0.25**2)))


def test_seed_is_local_and_mask_rate_is_configurable():
    np.random.seed(19)
    state = np.random.get_state()
    config = LimeConfig(num_samples=10000, mask_rate=0.75, seed=3)
    masks = sample_masks(10, config)
    np.testing.assert_array_equal(masks, sample_masks(10, config))
    assert not np.array_equal(masks, sample_masks(10, LimeConfig(num_samples=10000, seed=4)))
    assert 1 - masks[1:].mean() == pytest.approx(0.75, abs=0.01)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]
    assert masks[0].all()


def test_weighted_ridge_matches_augmented_least_squares():
    config = LimeConfig(num_samples=100, ridge_alpha=2.0)
    image = np.ones((1, 3, 1))
    segments = np.array([[0, 1, 2]])

    def predict(batch):
        x = batch[:, 0, :, 0]
        return (2 + x[:, 0] - 3*x[:, 1] + 4*x[:, 0]*x[:, 2])[:, None]

    result = fit_lime(image, segments, predict, 0, config)
    masks = sample_masks(3, config)
    x = np.column_stack([np.ones(100), masks])
    w = np.sqrt(locality_weights(masks, config.kernel_width))
    penalty = np.diag([0, *([np.sqrt(2)] * 3)])
    y = predict(masks[:, None, :, None]).ravel()
    expected = np.linalg.lstsq(np.vstack([x*w[:, None], penalty]),
                               np.r_[y*w, np.zeros(4)], rcond=None)[0]
    np.testing.assert_allclose(np.r_[result.intercept, result.coefficients], expected)


@pytest.mark.parametrize("kwargs", [{"num_samples": 1}, {"batch_size": 0},
    {"seed": -1}, {"seed": True}, {"num_samples": 3.5}, {"mask_rate": 1},
    {"mask_rate": float("nan")}, {"kernel_width": 0}, {"ridge_alpha": -1}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        LimeConfig(**kwargs)


@pytest.mark.parametrize("output", [np.zeros(3), np.zeros((3, 0)),
                                     np.full((3, 1), np.nan), np.zeros((1, 1))])
def test_bad_prediction_contract(output):
    with pytest.raises(ValueError, match="predict_logits"):
        fit_lime(np.ones((2, 2, 1)), np.zeros((2, 2), dtype=int),
                 lambda batch: output, 0, LimeConfig(num_samples=3))


def test_one_region_constant_model_and_batch_invariance():
    def predict(batch):
        return np.full((len(batch), 1), 2.0)
    image, segments = np.ones((2, 2, 1)), np.full((2, 2), -5)
    a = fit_lime(image, segments, predict, 0, LimeConfig(num_samples=20, batch_size=1))
    b = fit_lime(image, segments, predict, 0, LimeConfig(num_samples=20, batch_size=7))
    np.testing.assert_allclose(a.coefficients, [0], atol=1e-12)
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.intercept == pytest.approx(2)


def test_nonconstant_model_is_invariant_to_batch_size():
    image = np.arange(12, dtype=float).reshape(2, 2, 3)
    segments = np.array([[0, 1], [2, 3]])

    def predict(batch):
        return np.sin(batch.sum(axis=(1, 2, 3)))[:, None]

    a = fit_lime(image, segments, predict, 0, LimeConfig(num_samples=37, batch_size=1))
    b = fit_lime(image, segments, predict, 0, LimeConfig(num_samples=37, batch_size=8))
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.intercept == b.intercept
    assert a.forward_samples == b.forward_samples == 37


@pytest.mark.parametrize("image, segments, target", [
    (np.ones((2, 2)), np.zeros((2, 2), dtype=int), 0),
    (np.ones((2, 2, 1), dtype=int), np.zeros((2, 2), dtype=int), 0),
    (np.ones((2, 2, 1)), np.zeros((1, 2), dtype=int), 0),
    (np.ones((2, 2, 1)), np.zeros((2, 2)), 0),
    (np.ones((2, 2, 1)), np.zeros((2, 2), dtype=int), -1),
])
def test_invalid_inputs_fail_before_model_call(image, segments, target):
    def predict(batch):
        pytest.fail("invalid input must not consume forwards")

    with pytest.raises(ValueError):
        fit_lime(image, segments, predict, target)


def test_slic_accepts_rgb_and_rejects_normalized_input():
    rgb = np.random.default_rng(0).random((32, 32, 3))
    labels = slic_segments(rgb, n_segments=8)
    assert labels.shape == (32, 32)
    assert labels.min() == 0
    np.testing.assert_array_equal(labels, slic_segments(rgb, n_segments=8))
    with pytest.raises(ValueError):
        slic_segments(rgb - 0.5)
