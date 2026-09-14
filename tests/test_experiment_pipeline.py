"""Analytic evaluation and real torch adapter tests, without course checkpoints."""
import numpy as np
import pytest

from evaluation.perturbation import (
    input_stability,
    insertion_deletion,
    localization_metrics,
    region_attribution_map,
    seed_stability,
)


def test_auc_endpoints_actual_fractions_and_separate_budget():
    image = np.array([[[3.0], [2.0], [1.0]]])
    segments = np.array([[10, 20, 30]])
    seen = []

    def predict(batch):
        seen.extend(batch)
        return batch.sum(axis=(1, 2, 3))[:, None]

    result = insertion_deletion(image, segments, np.array([3., 2., 1.]), predict, 0,
                                batch_size=2, steps=20)
    np.testing.assert_allclose(result["fractions"], [0, 1/3, 2/3, 1])
    assert result["insertion"] == [0, 3, 5, 6]
    assert result["deletion"] == [6, 3, 1, 0]
    assert result["insertion_auc"] == pytest.approx(11/3)
    assert result["deletion_auc"] == pytest.approx(7/3)
    assert result["evaluation_forward_samples"] == len(seen) == 8
    reverse = insertion_deletion(image, segments, np.array([1., 2., 3.]), predict, 0)
    assert result["insertion_auc"] > reverse["insertion_auc"]
    assert result["deletion_auc"] < reverse["deletion_auc"]


def test_signed_scores_and_ties_have_deterministic_order():
    image = np.array([[[2.0], [-3.0]]])
    scores = np.array([-1.0, -1.0])
    result = insertion_deletion(image, np.array([[5, 9]]), scores,
                                lambda b: b.sum(axis=(1, 2, 3))[:, None], 0)
    assert result["insertion"] == [0, 2, -1]


def test_stability_known_rankings_and_constant_case():
    assert seed_stability([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])["spearman"] == pytest.approx(1)
    reversed_scores = seed_stability([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert reversed_scores["spearman"] == pytest.approx(-1)
    assert reversed_scores["top20_jaccard"] == 0
    assert seed_stability([1, 1], [1, 2])["spearman"] is None


def test_input_stability_signed_vectors():
    result = input_stability(
        np.array([3.0, 4.0]),
        np.array([[3.0, 4.0], [0.0, 5.0]]),
    )
    assert result["max_sensitivity"] == pytest.approx(np.sqrt(10) / 5)
    assert result["cosine_similarity"] == pytest.approx(0.9)
    assert result["stability_repeats"] == 2
    assert not result["zero_reference"]
    assert result["zero_perturbed"] == 0


def test_input_stability_zero_norm_diagnostics():
    result = input_stability(np.zeros(2), np.ones((3, 2)))
    assert result["max_sensitivity"] is None
    assert result["cosine_similarity"] is None
    assert result["zero_reference"]
    result = input_stability(np.ones(2), np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert result["max_sensitivity"] == pytest.approx(1.0)
    assert result["cosine_similarity"] is None
    assert result["zero_perturbed"] == 1


def test_region_map_conserves_signed_contributions():
    segments = np.array([[4, 4, 9], [4, 9, 9]])
    result = region_attribution_map(segments, np.array([6.0, -3.0]))
    np.testing.assert_allclose(result, [[2, 2, -1], [2, -1, -1]])
    assert result[segments == 4].sum() == pytest.approx(6)
    assert result[segments == 9].sum() == pytest.approx(-3)


def test_localization_metrics_positive_energy_and_pointing():
    attribution = np.array([[3.0, 1.0], [-7.0, 2.0]])
    mask = np.array([[True, False], [True, False]])
    result = localization_metrics(attribution, mask)
    assert result["localization_energy"] == pytest.approx(0.5)
    assert result["pointing_game"] == 1
    assert not result["zero_positive_energy"]
    zero = localization_metrics(-np.ones((2, 2)), mask)
    assert zero == {
        "localization_energy": 0.0,
        "pointing_game": 0,
        "zero_positive_energy": True,
    }


def test_torch_adapter_eval_layout_and_counts():
    torch = pytest.importorskip("torch")
    from models.inference import LogitPredictor
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    image = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    predictor = LogitPredictor(model, "cpu")
    with torch.no_grad():
        expected = model(torch.from_numpy(image.transpose(0, 3, 1, 2).copy())).numpy()
    np.testing.assert_allclose(predictor(image), expected)
    assert not model.training
    assert predictor.samples == 2 and predictor.batches == 1


def test_balanced_selection_reproducible_and_not_input_order_dependent():
    pytest.importorskip("torch")
    from experiments.perturbation_pilot import balanced_subset
    rows = [{"id": str(i), "target": i % 2} for i in range(30)]
    a = balanced_subset(rows, 5, 0)
    assert a == balanced_subset(rows[::-1], 5, 0)
    assert sum(r["target"] == 1 for r in a) == 5
    with pytest.raises(ValueError):
        balanced_subset(rows, 16, 0)


def test_all_joint_correct_selection_keeps_every_row_in_id_order():
    pytest.importorskip("torch")
    from experiments.perturbation_pilot import select_samples
    rows = [{"id": "b", "target": 1}, {"id": "a", "target": 0}]
    assert select_samples(rows, {"selection": "all_joint_correct"}) == rows[::-1]


def test_seeded_subset_reproducible_and_validated():
    pytest.importorskip("torch")
    from experiments.perturbation_pilot import seeded_subset, select_samples
    rows = [{"id": str(i), "target": i % 3} for i in range(30)]
    selected = seeded_subset(rows, 7, 4)
    assert selected == seeded_subset(rows[::-1], 7, 4)
    assert selected == select_samples(rows, {"selection": "seeded_subset",
                                             "sample_count": 7, "selection_seed": 4})
    with pytest.raises(ValueError):
        seeded_subset(rows, 31, 0)


def test_resume_rejects_source_and_data_changes(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from experiments import perturbation_pilot as pilot
    root = tmp_path/"repository"
    (root/"models").mkdir(parents=True)
    (root/"images").mkdir()
    image = root/"images/CHNCXR_0001_0.png"
    image.write_bytes(b"fixture: fingerprint only; no image decoding")
    checkpoint = root/"models/fixture.pt"
    checkpoint.write_bytes(b"fixture: no checkpoint loading")
    source = root/"models/fixture.py"
    source.write_text("# original", encoding="utf-8")
    monkeypatch.setattr(pilot, "ROOT", root)
    config = {"data_root": "images", "models": {"resnet50": "models/fixture.pt"}, "device": "cpu"}
    output = tmp_path/"run"
    first = pilot.freeze_run(config, output, False)
    assert pilot.freeze_run(config, output, True)["fingerprint"] == first["fingerprint"]
    with pytest.raises(ValueError, match="Output exists"):
        pilot.freeze_run(config, output, False)
    source.write_text("# modified", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        pilot.freeze_run(config, output, True)
    source.write_text("# original", encoding="utf-8")
    image.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="changed"):
        pilot.freeze_run(config, output, True)
