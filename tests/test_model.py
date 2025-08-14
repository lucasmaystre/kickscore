import json
import pickle
import random

import numpy as np
import pytest

import kickscore as ks


def test_json_example(testcase_path: str):
    """Verify a test case described in JSON format."""
    with open(testcase_path) as f:
        raw = "".join(line for line in f if not line.startswith("//"))
    data = json.loads(raw)
    model_class = getattr(ks, data["model_class"])
    model = model_class(**data.get("model_args", {}))
    for item in data["items"]:
        kernel_class = getattr(ks.kernel, item["kernel_class"])
        kernel = kernel_class(**item["kernel_args"])
        model.add_item(item["name"], kernel=kernel)
    for obs in data["observations"]:
        model.observe(**obs)
    model.fit(**data.get("fit_args", {}))
    for name, scores in data["scores"].items():
        _, mean, var = model.item[name].scores
        assert np.allclose(scores["mean"], mean, rtol=1e-3)
        assert np.allclose(scores["var"], var, rtol=1e-3)
    assert np.allclose(model.log_likelihood, data["log_likelihood"], rtol=1e-3)


@pytest.mark.parametrize("model", [ks.BinaryModel(), ks.TernaryModel()])
def test_chronological_order(model: ks.model.Model):
    """Observations can only be added in chronological order."""
    model.add_item("x", kernel=ks.kernel.Constant(1.0))
    model.observe(winners=["x"], losers=[], t=1.0)
    with pytest.raises(ValueError):
        model.observe(winners=["x"], losers=[], t=0.0)


def test_damping():
    """Damping should work on a simple example."""
    kernel = ks.kernel.Constant(1.0)
    model = ks.BinaryModel()
    for x in ["A", "B", "C", "D"]:
        model.add_item(x, kernel=kernel)
    model.observe(winners=["C", "D"], losers=["A", "B"], t=0.0)
    model.observe(winners=["A", "B"], losers=["C", "D"], t=0.0)
    model.observe(winners=["A", "B"], losers=["C", "D"], t=0.0)
    # Without damping, this simple example diverges.
    assert not model.fit(max_iter=20)
    # However, a little bit of damping is enough to make it converge.
    assert model.fit(max_iter=20, lr=0.8)


def test_add_item_twice():
    """Item with same name cannot be added again."""
    kernel = ks.kernel.Constant(1.0)
    model = ks.BinaryModel()
    model.add_item("x", kernel)
    with pytest.raises(ValueError):
        model.add_item("x", kernel)


def test_saving():
    """Serializing a large(-ish) model with pickle should work."""
    random.seed(0)
    kernel = ks.kernel.Constant(1.0)
    model = ks.BinaryModel()
    for i in range(100):
        model.add_item(f"team{i}", kernel)
    for _ in range(500):
        i, j = random.sample(list(model.item.keys()), 2)
        model.observe(winners=[i], losers=[j], t=0.0)
    # Serialize & unserialize.
    data = pickle.dumps(model)
    model2 = pickle.loads(data)
    assert model2.item.keys() == model.item.keys()
    for obs, obs2 in zip(model.observations, model2.observations):
        assert obs.t == obs2.t
