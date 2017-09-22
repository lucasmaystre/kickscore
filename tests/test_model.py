import json
import kickscore as ks
import numpy as np


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
        assert np.allclose(scores["mean"], mean)
        assert np.allclose(scores["var"], var)
    assert np.allclose(model.log_likelihood, data["log_likelihood"])
