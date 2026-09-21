"""Analytic and interface checks without checkpoints or dataset access."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from attribution.perturbation.ablation import AblationConfig, fit_ablation


@pytest.mark.parametrize("batch_size", [1, 3, 32])
@pytest.mark.parametrize("cached", [False, True])
def test_signed_effects_independent_masks_and_exact_budget(batch_size, cached):
    image = np.array([[[2., 1.], [-3., -1.]], [[1., 2.], [4., 1.]]], dtype=np.float32)
    segments = np.array([[10, -5], [10, 30]])
    before, labels_before = image.copy(), segments.copy()
    seen = []

    def predict(batch):
        assert batch.dtype == image.dtype
        seen.append(batch.copy())
        y = 7 + batch.sum(axis=(1, 2, 3))
        return np.column_stack([-y, y])

    result = fit_ablation(image, segments, predict, 1, AblationConfig(batch_size),
                          original_logit=14. if cached else None)
    np.testing.assert_array_equal(result.region_ids, [-5, 10, 30])
    np.testing.assert_array_equal(result.coefficients, [-4, 6, 5])
    np.testing.assert_array_equal(result.ablated_logits, [18, 8, 9])
    assert result.target == 1
    assert result.original_logit == 14
    assert result.used_cached_original is cached
    total = 3 if cached else 4
    assert result.forward_samples == sum(len(b) for b in seen) == total
    assert result.forward_batches == len(seen) == (total + batch_size - 1) // batch_size
    assert all(len(b) <= batch_size for b in seen)
    actual = np.concatenate(seen)
    if not cached:
        np.testing.assert_array_equal(actual[0], image)
        actual = actual[1:]
    for region, perturbed in zip(result.region_ids, actual):
        expected = image.copy()
        expected[segments == region] = 0
        np.testing.assert_array_equal(perturbed, expected)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(segments, labels_before)


def test_interaction_is_full_context_deletion_not_sequential_or_shapley():
    def predict(batch):
        x, y = batch[:, 0, 0, 0], batch[:, 0, 1, 0]
        return (7 + 2*x - 3*y + 5*x*y)[:, None]

    a = fit_ablation(np.ones((1, 2, 1)), np.array([[0, 1]]), predict, 0)
    b = fit_ablation(np.ones((1, 2, 1)), np.array([[0, 1]]), predict, 0,
                     AblationConfig(batch_size=1))
    assert a.original_logit == 11
    np.testing.assert_array_equal(a.ablated_logits, [4, 9])
    np.testing.assert_array_equal(a.coefficients, [7, 2])
    np.testing.assert_array_equal(a.coefficients, b.coefficients)


def test_one_region_and_zero_cached_logit():
    seen = []

    def predict(batch):
        seen.extend(batch.copy())
        return (batch.sum(axis=(1, 2, 3)) - 4)[:, None]

    result = fit_ablation(np.ones((2, 2, 1)), np.full((2, 2), -5), predict, 0,
                          original_logit=0.)
    np.testing.assert_array_equal(result.coefficients, [4])
    np.testing.assert_array_equal(seen, np.zeros((1, 2, 2, 1)))
    assert result.forward_samples == 1
    assert result.used_cached_original


def test_constant_model_is_zero_and_does_not_use_random_state():
    state = np.random.get_state()
    args = (np.ones((1, 2, 1)), np.array([[0, 1]]),
            lambda batch: np.full((len(batch), 1), 2.), 0)
    a, b = fit_ablation(*args), fit_ablation(*args)
    np.testing.assert_array_equal(a.coefficients, [0, 0])
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_bad_config(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        AblationConfig(batch_size)


@pytest.mark.parametrize("kwargs", [
    {"image": np.ones((2, 2))}, {"image": np.ones((2, 2, 1), dtype=int)},
    {"image": np.full((2, 2, 1), np.nan)}, {"image": np.ones((0, 2, 1))},
    {"segments": np.zeros((1, 2), dtype=int)}, {"segments": np.zeros((2, 2))},
    {"target": -1}, {"target": True}, {"target": 0.5},
    {"original_logit": np.nan}, {"original_logit": np.inf},
    {"original_logit": True}, {"original_logit": [1.]},
])
def test_invalid_inputs_do_not_consume_forwards(kwargs):
    def predict(batch):
        pytest.fail("invalid input must fail before inference")

    args = dict(image=np.ones((2, 2, 1)), segments=np.zeros((2, 2), dtype=int),
                predict_logits=predict, target=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        fit_ablation(**args)


@pytest.mark.parametrize("output", [
    np.zeros(2), np.zeros((2, 0)), np.full((2, 1), np.nan),
    np.zeros((1, 1)), np.full((2, 1), 1j), np.full((2, 1), 'bad'),
])
def test_bad_prediction_contract(output):
    with pytest.raises(ValueError, match="predict_logits"):
        fit_ablation(np.ones((2, 2, 1)), np.zeros((2, 2), dtype=int),
                     lambda batch: output, 0)


def test_changed_class_count_is_rejected():
    calls = []

    def predict(batch):
        calls.append(1)
        return np.zeros((len(batch), len(calls)))

    with pytest.raises(ValueError, match="class count"):
        fit_ablation(np.ones((1, 2, 1)), np.array([[0, 1]]), predict, 0,
                     AblationConfig(batch_size=1))


@pytest.mark.parametrize("cached", [False, True])
def test_dry_run_budget(cached):
    cmd = [sys.executable, '-B', '-m', 'experiments.ablation_baseline',
           '--dry-run', '--num-regions', '32']
    if cached:
        cmd.append('--cached-original')
    run = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=True)
    result = json.loads(run.stdout)
    assert result['forward_samples_per_image'] == (32 if cached else 33)
    assert result['forward_batches_per_image'] == (1 if cached else 2)
    assert result['executes_attribution'] is False
