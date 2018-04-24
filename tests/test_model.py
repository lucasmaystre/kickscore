import json
import kickscore as ks
import numpy as np
import pytest


def test_json_example(testcase_path):
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
    model.fit()
    for name, scores in data["scores"].items():
        _, mean, var = model.item[name].scores
        assert np.allclose(scores["mean"], mean, rtol=1e-3)
        assert np.allclose(scores["var"], var, rtol=1e-3)
    assert np.allclose(model.log_likelihood, data["log_likelihood"], rtol=1e-3)


@pytest.mark.parametrize("model", [ks.BinaryModel(), ks.TernaryModel()])
def test_chronological_order(model):
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
    assert model.fit(max_iter=20, damping=0.8)
