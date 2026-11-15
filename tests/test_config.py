from __future__ import annotations

import dataclasses

import pytest

from uapd.config import Config, ConfigError


def test_defaults_match_v1_constants():
    config = Config()
    assert config.source.live_url.startswith("https://www.youtube.com/watch?v=")
    assert config.frame.crop_left == 165
    assert config.frame.crop_right == 176
    assert config.regions.cycle_frames == 120
    assert config.regions.thresholds["lf"] == 0.99
    assert config.candidates.min_contrast == 23
    assert config.candidates.max_bg_brightness == 23
    assert config.capture.post_detection_seconds == 2.1


def test_profile_overrides_only_what_it_names(tmp_path):
    profile = tmp_path / "profile.toml"
    profile.write_text("[candidates]\nmin_contrast = 15\n\n[regions.thresholds]\nspace = 0.9\n")
    config = Config.load(profile)
    assert config.candidates.min_contrast == 15
    assert config.candidates.min_area == 25
    assert config.regions.thresholds["space"] == 0.9
    assert config.regions.thresholds["earth"] == 0.75


def test_int_is_accepted_for_float_field():
    config = Config.from_dict({"capture": {"buffer_seconds": 4}})
    assert config.capture.buffer_seconds == 4.0
    assert isinstance(config.capture.buffer_seconds, float)


@pytest.mark.parametrize(
    "data",
    [
        {"candidates": {"min_contrst": 15}},
        {"nonsense": {}},
        {"regions": {"thresholds": {"ufo": 0.5}}},
    ],
)
def test_unknown_keys_are_rejected(data):
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"candidates": {"min_contrast": "high"}},
        {"candidates": {"min_contrast": 1.5}},
        {"candidates": {"min_contrast": True}},
        {"source": {"live_url": 5}},
        {"candidates": {"filter_lf": 1}},
        {"candidates": 5},
    ],
)
def test_wrong_types_are_rejected(data):
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().candidates.min_contrast = 1


def test_fingerprint_tracks_values():
    base = Config()
    assert base.fingerprint() == Config().fingerprint()
    changed = Config.from_dict({"candidates": {"min_contrast": 22}})
    assert changed.fingerprint() != base.fingerprint()
