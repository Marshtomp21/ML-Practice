"""OrdShap estimator and image patch-lattice adapter tests."""
import numpy as np
import pytest

from attribution.game_theoretic.ordshap import (
    OrdShapConfig,
    compose_patches,
    estimate_ordshap,
    fit_ordshap,
    patch_labels,
    split_patches,
)


def vector_evaluator(sample, model):
    def evaluate(masks, perms):
        present = np.where(masks, sample, 0.0)
        return model(np.take_along_axis(present, perms, axis=1))
    return evaluate


def exact_config(d):
    return OrdShapConfig(num_coalitions=2 ** d, num_permutations=10 ** 4,
                         full_permutations=10 ** 4)


def test_position_invariant_model_has_shapley_vi_and_zero_pi():
    sample = np.array([1.0, -2.0, 0.5, 3.0])
    # The model sums values regardless of slot, so content alone matters.
    out = estimate_ordshap(vector_evaluator(sample, lambda x: x.sum(axis=1)), 4, exact_config(4))
    np.testing.assert_allclose(out["value_importance"], sample, atol=1e-12)
    np.testing.assert_allclose(out["position_importance"], 0.0, atol=1e-12)


def test_linear_slot_weights_give_linear_position_slope():
    sample = np.array([2.0, -1.0, 0.5, 1.5])
    slots = np.arange(1, 5, dtype=float)  # weight grows by 1 per slot
    out = estimate_ordshap(vector_evaluator(sample, lambda x: x @ slots), 4, exact_config(4))
    # E_sigma contribution = x_i * mean slot weight; slope over slots = x_i.
    np.testing.assert_allclose(out["value_importance"], sample * slots.mean(), atol=1e-12)
    np.testing.assert_allclose(out["position_importance"], sample, atol=1e-12)
    assert abs(out["efficiency_residual"]) < 1e-12


def test_patch_lattice_roundtrip_and_permutation_moves_content():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8, 3)).astype(np.float32)
    patches = split_patches(image, 2)
    identity = np.arange(4)[None]
    full = np.ones((1, 4), dtype=bool)
    np.testing.assert_array_equal(compose_patches(patches, 0.0, full, identity, 2)[0], image)
    swapped = compose_patches(patches, 0.0, full, np.array([[3, 1, 2, 0]]), 2)[0]
    np.testing.assert_array_equal(swapped[:4, :4], image[4:, 4:])
    labels = patch_labels(8, 8, 2)
    assert labels[0, 7] == 1 and labels[7, 0] == 2
    absent = compose_patches(patches, np.zeros(3, np.float32),
                             np.array([[True, False, True, True]]), identity, 2)[0]
    assert np.all(absent[:4, 4:] == 0)


def test_fit_ordshap_counts_forwards_and_rejects_nonuniform_baseline():
    rng = np.random.default_rng(1)
    image = rng.normal(size=(8, 8, 3))
    calls = []

    def predict(batch):
        calls.append(len(batch))
        # Top-left slot matters more: a position-dependent toy classifier.
        score = batch[:, :4, :4].mean(axis=(1, 2, 3)) * 2 + batch.mean(axis=(1, 2, 3))
        return np.stack([score, -score], axis=1)

    config = OrdShapConfig(num_coalitions=8, num_permutations=3, full_permutations=5,
                           batch_size=4, seed=2)
    result = fit_ordshap(image, 2, predict, 0, config)
    assert result.forward_samples == sum(calls) == 1 + 1 + 5 + 6 * 3
    assert result.position_importance_2d.shape == (4, 2)
    assert abs(result.efficiency_residual) < 1e-9
    again = fit_ordshap(image, 2, predict, 0, config)
    np.testing.assert_array_equal(result.value_importance, again.value_importance)
    with pytest.raises(ValueError, match="uniform"):
        fit_ordshap(image, 2, predict, 0, config, baseline=rng.normal(size=image.shape))
