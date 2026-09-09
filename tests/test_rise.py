"""Synthetic numerical checks: no checkpoints, dataset access or saved maps."""

import numpy as np
import pytest

from attribution.perturbation.rise import (
    RiseConfig,
    explain_rise,
    generate_rise_masks,
    region_scores_from_saliency,
)


def test_matches_definition_and_counts_exact_forward_budget():
    config = RiseConfig(num_masks=17, batch_size=5, grid_size=3, mask_rate=0.4, seed=11)
    image = np.arange(1, 25, dtype=np.float64).reshape(4, 3, 2) / 10
    masks = generate_rise_masks(image.shape[:2], config)
    seen = []

    def predict(batch):
        seen.append(batch.copy())
        target_values = 2.0 + batch[..., 0].sum(axis=(1, 2))
        return np.column_stack([-target_values, target_values])

    original = image.copy()
    result = explain_rise(image, predict, 1, config)
    values = 2.0 + (image[None, ..., 0] * masks).sum(axis=(1, 2))
    expected = np.einsum("b,bhw->hw", values, masks) / (
        config.num_masks * config.keep_probability
    )

    np.testing.assert_allclose(result.saliency, expected, rtol=1e-12, atol=1e-12)
    assert result.mean_target_logit == pytest.approx(values.mean())
    assert result.forward_samples == sum(len(batch) for batch in seen) == 17
    assert result.forward_batches == len(seen) == 4
    assert result.target == 1
    assert result.keep_probability == pytest.approx(0.6)
    np.testing.assert_array_equal(
        np.concatenate(seen), image[None, ...] * masks[..., None]
    )
    np.testing.assert_array_equal(image, original)


def test_masks_are_reproducible_soft_and_do_not_touch_global_rng():
    np.random.seed(19)
    state = np.random.get_state()
    config = RiseConfig(num_masks=500, grid_size=4, mask_rate=0.75, seed=3)
    masks = generate_rise_masks((9, 13), config)

    np.testing.assert_array_equal(masks, generate_rise_masks((9, 13), config))
    assert not np.array_equal(
        masks,
        generate_rise_masks(
            (9, 13), RiseConfig(num_masks=500, grid_size=4, mask_rate=0.75, seed=4)
        ),
    )
    assert masks.shape == (500, 9, 13)
    assert masks.dtype == np.float32
    assert np.isfinite(masks).all()
    assert 0 <= masks.min() <= masks.max() <= 1
    assert np.any((masks > 0) & (masks < 1))
    assert masks.mean() == pytest.approx(config.keep_probability, abs=0.02)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]


def test_no_unperturbed_mask_is_forced_into_sampling():
    config = RiseConfig(num_masks=8, grid_size=3, mask_rate=0.5, seed=0)
    masks = generate_rise_masks((7, 7), config)
    assert not np.all(masks[0] == 1)


def test_batch_size_does_not_change_masks_or_saliency():
    image = np.arange(30, dtype=float).reshape(3, 5, 2) / 7

    def predict(batch):
        value = np.cos(batch.sum(axis=(1, 2, 3)))
        return value[:, None]

    a = explain_rise(
        image,
        predict,
        0,
        RiseConfig(num_masks=23, batch_size=1, grid_size=4, mask_rate=0.25, seed=8),
    )
    b = explain_rise(
        image,
        predict,
        0,
        RiseConfig(num_masks=23, batch_size=7, grid_size=4, mask_rate=0.25, seed=8),
    )
    np.testing.assert_allclose(a.saliency, b.saliency, rtol=1e-15, atol=1e-15)
    assert a.mean_target_logit == pytest.approx(b.mean_target_logit)
    assert a.realized_grid_keep_probability == b.realized_grid_keep_probability
    assert a.realized_pixel_keep_probability == b.realized_pixel_keep_probability
    assert a.forward_samples == b.forward_samples == 23
    assert a.forward_batches == 23
    assert b.forward_batches == 4


def test_result_diagnostics_match_generated_masks():
    config = RiseConfig(num_masks=29, batch_size=6, grid_size=3, seed=2)
    masks = generate_rise_masks((5, 4), config)
    result = explain_rise(
        np.ones((5, 4, 1)), lambda batch: np.ones((len(batch), 1)), 0, config
    )
    np.testing.assert_allclose(
        result.saliency,
        masks.sum(axis=0) / (config.num_masks * config.keep_probability),
    )
    assert result.realized_pixel_keep_probability == pytest.approx(masks.mean())
    assert 0 <= result.realized_grid_keep_probability <= 1
    assert 1 <= result.unique_low_resolution_masks <= config.num_masks


def test_region_scores_sum_signed_pixel_saliency_for_noncontiguous_labels():
    saliency = np.array([[1.0, -2.0], [3.0, 4.0]])
    segments = np.array([[10, 20], [10, 30]])
    region_ids, scores = region_scores_from_saliency(saliency, segments)
    np.testing.assert_array_equal(region_ids, [10, 20, 30])
    np.testing.assert_allclose(scores, [4, -2, 4])
    assert scores.sum() == pytest.approx(saliency.sum())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_masks": 0},
        {"batch_size": 0},
        {"grid_size": 1},
        {"seed": -1},
        {"seed": True},
        {"num_masks": 3.5},
        {"mask_rate": 0},
        {"mask_rate": 1},
        {"mask_rate": float("nan")},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        RiseConfig(**kwargs)


@pytest.mark.parametrize(
    "output",
    [
        np.zeros(3),
        np.zeros((3, 0)),
        np.full((3, 1), np.nan),
        np.zeros((1, 1)),
        np.full((3, 1), "bad"),
    ],
)
def test_bad_prediction_contract(output):
    with pytest.raises(ValueError, match="predict_logits"):
        explain_rise(
            np.ones((2, 2, 1)),
            lambda batch: output,
            0,
            RiseConfig(num_masks=3, batch_size=3),
        )


def test_prediction_class_count_cannot_change_between_batches():
    calls = 0

    def predict(batch):
        nonlocal calls
        calls += 1
        return np.zeros((len(batch), calls + 1))

    with pytest.raises(ValueError, match="class count"):
        explain_rise(
            np.ones((2, 2, 1)),
            predict,
            0,
            RiseConfig(num_masks=3, batch_size=2),
        )


@pytest.mark.parametrize(
    "image, target",
    [
        (np.ones((2, 2)), 0),
        (np.ones((2, 2, 1), dtype=int), 0),
        (np.full((2, 2, 1), np.nan), 0),
        (np.ones((2, 2, 1)), -1),
        (np.ones((2, 2, 1)), True),
    ],
)
def test_invalid_inputs_fail_before_model_call(image, target):
    def predict(batch):
        pytest.fail("invalid input must not consume forwards")

    with pytest.raises(ValueError):
        explain_rise(image, predict, target)


@pytest.mark.parametrize("shape", [(4,), (4, 5, 6), (0, 4), (True, 4), (4.5, 4)])
def test_invalid_mask_shape(shape):
    with pytest.raises(ValueError, match="image_shape"):
        generate_rise_masks(shape)


@pytest.mark.parametrize(
    "saliency, segments",
    [
        (np.ones((2, 2, 1)), np.zeros((2, 2), dtype=int)),
        (np.ones((2, 2), dtype=int), np.zeros((2, 2), dtype=int)),
        (np.full((2, 2), np.nan), np.zeros((2, 2), dtype=int)),
        (np.ones((2, 2)), np.zeros((1, 2), dtype=int)),
        (np.ones((2, 2)), np.zeros((2, 2))),
    ],
)
def test_invalid_region_aggregation(saliency, segments):
    with pytest.raises(ValueError):
        region_scores_from_saliency(saliency, segments)
