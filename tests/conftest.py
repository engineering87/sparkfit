"""Shared fixtures: keep every test independent of the developer's own setup."""
import pytest

import sparkfit as sf


@pytest.fixture(autouse=True)
def _isolate_user_config(tmp_path, monkeypatch):
    """Ignore the real ~/.sparkfitrc and any SPARKFIT_* variables in the shell.

    A calibrated config on the machine running the tests (e.g. `efficiency = 0.4`)
    would otherwise change the defaults the tests assert against.
    """
    for name in list(sf._CONFIG_KEYS.values()):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(sf.CONFIG_ENV, str(tmp_path / "no-sparkfitrc"))
    monkeypatch.setattr(sf, "_CONFIG", {})
