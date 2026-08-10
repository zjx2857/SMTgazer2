from unittest.mock import patch

from ConfigSpace import ConfigurationSpace, Float

from smac.model.xg.xg import XG


def test_lightgbm_uses_gpu_backend(monkeypatch):
    monkeypatch.delenv("SMTGAZER_LIGHTGBM_DEVICE_TYPE", raising=False)
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x", (0.0, 1.0)))

    with patch("smac.model.xg.xg.lgb.LGBMRegressor") as regressor:
        XG(configspace=configspace)

    assert regressor.call_args.kwargs["device_type"] == "gpu"


def test_lightgbm_uses_configured_cuda_backend(monkeypatch):
    monkeypatch.setenv("SMTGAZER_LIGHTGBM_DEVICE_TYPE", "cuda")
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x", (0.0, 1.0)))

    with patch("smac.model.xg.xg.lgb.LGBMRegressor") as regressor:
        XG(configspace=configspace)

    assert regressor.call_args.kwargs["device_type"] == "cuda"
