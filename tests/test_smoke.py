"""CPU smoke tests (no downloads): pytest -q"""
import numpy as np
import pytest
import torch

from tabpfnseg.data.datasets import SyntheticDataset
from tabpfnseg.data.episodes import make_episodes
from tabpfnseg.features.builder import FeatureSpec, jbu_upsample, select_context
from tabpfnseg.methods.feature_segmenter import prior_correct
from tabpfnseg.run import main


def test_episodes_no_query_in_support():
    ds = SyntheticDataset()
    for shots in (1, 5):
        for e in make_episodes(ds, shots, None, seed=0, dedup_hamming=2):
            assert e.query[0] not in [s[0] for s in e.supports]
            assert len({s[0] for s in e.supports}) == shots
            assert all(s[0].split("/")[0] == e.cls for s in e.supports)


def test_episodes_deterministic():
    ds = SyntheticDataset()
    a = make_episodes(ds, 1, 20, seed=3)
    b = make_episodes(ds, 1, 20, seed=3)
    assert [(e.query, e.supports) for e in a] == [(e.query, e.supports) for e in b]


def test_feature_spec_dims():
    assert FeatureSpec.parse("layers=4,8,12,16,20,24").dim(1024) == 6144
    assert FeatureSpec.parse("layers=24;pools=1,3,5,9,17,g").dim(1024) == 6144
    assert FeatureSpec.parse("layers=24;rgb=1").dim(1024) == 1030


def test_jbu_shape_and_constant():
    f = torch.ones(5, 4, 4)
    g = torch.rand(3, 16, 16)
    up = jbu_upsample(f, g, (16, 16))
    assert up.shape == (5, 16, 16) and torch.allclose(up, torch.ones_like(up), atol=1e-5)


def test_prior_correction_identity_and_direction():
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(prior_correct(p, 0.3, 0.3), p)
    assert (prior_correct(p, 0.5, 0.1) < p).all()  # context over-represents fg -> push down


def test_balanced_sampling():
    rng = np.random.default_rng(0)
    fr = np.zeros((40, 40)); fr[:8] = 1
    idx, sampled, true = select_context([fr], "balanced", 200, rng)
    assert abs(sampled - 0.5) < 0.01 and abs(true - 0.2) < 0.01


def test_leakage_guard_blocks_tuning_on_test():
    with pytest.raises(SystemExit):
        main(["--dataset", "synthetic", "--split", "test", "--purpose", "tune", "--device", "cpu"])


@pytest.mark.parametrize("extra", [
    ["--method", "proto"],
    ["--method", "knn", "--features", "layers=-1,2;pools=1,3,g;rgb=1"],
    ["--method", "logreg", "--res", "x4", "--up", "jbu", "--sampling", "boundary", "--max_ctx", "1500", "--pca", "16"],
    ["--method", "oracle_grid", "--res", "x2"],
])
def test_end_to_end(tmp_path, extra):
    s = main(["--dataset", "synthetic", "--backbone", "dummy", "--img_size", "160", "--device", "cpu",
              "--limit", "8", "--out", str(tmp_path)] + extra)
    assert 0 <= s["mIoU"] <= 100 and s["n_episodes"] == 8
