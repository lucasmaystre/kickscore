import glob
import os.path

from pytest import Metafunc

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def pytest_generate_tests(metafunc: Metafunc):
    if "testcase_path" in metafunc.fixturenames:
        pattern = os.path.join(DATA_ROOT, "testcase-*.json")
        paths: list[str] = glob.glob(pattern)
        metafunc.parametrize("testcase_path", paths)
