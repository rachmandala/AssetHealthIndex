"""Unit tests for configuration loading (src/config/settings.py)."""

from __future__ import annotations

import pytest

from src.config.settings import AppSettings, get_settings, reload_settings


def test_get_settings_loads_successfully() -> None:
    settings = get_settings()
    assert isinstance(settings, AppSettings)
    assert settings.project.name == "Asset Health Index"


def test_get_settings_is_cached() -> None:
    first = get_settings()
    second = get_settings()
    assert first is second


def test_bp_network_defaults() -> None:
    settings = get_settings()
    assert settings.bp_network.hidden_layer_sizes == [20]
    assert settings.bp_network.max_iter == 1000
    assert settings.bp_network.random_state == 42
    assert "Healthy" in settings.bp_network.classes
    assert "Fault" in settings.bp_network.classes


def test_health_index_b_parameter() -> None:
    settings = get_settings()
    assert settings.health_index.b == 0.5


def test_thresholds_ordering() -> None:
    settings = get_settings()
    t = settings.thresholds
    assert t.healthy > t.subhealthy > t.abnormal


def test_model_split_defaults() -> None:
    settings = get_settings()
    assert settings.model.train_split == pytest.approx(0.60)
    assert settings.model.test_split == pytest.approx(0.40)


def test_reload_settings_returns_new_but_equal_instance() -> None:
    original = get_settings()
    reloaded = reload_settings()
    assert reloaded is not original
    assert reloaded.project.name == original.project.name


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        get_settings.cache_clear()
        get_settings(config_path="config/does_not_exist.yaml")
    # Restore cache for subsequent tests.
    reload_settings()
