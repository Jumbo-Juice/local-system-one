"""The `python -m system_one check` self-test passes on the mock backend."""

from system_one import load_config, make_engine
from system_one.__main__ import check
from system_one.config import REPO_ROOT


def test_check_passes_on_mock(capsys):
    assert check(make_engine(load_config(REPO_ROOT / "config" / "mock.toml"))) == 0
    assert capsys.readouterr().out.strip().endswith("OK")
