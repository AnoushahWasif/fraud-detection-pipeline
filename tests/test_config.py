import pytest

from fraud_pipeline import config


def write(tmp_path, text):
    path = tmp_path / "pipeline.yaml"
    path.write_text(text)
    return path


def test_the_real_config_loads():
    cfg, digest = config.load()
    assert cfg.split.train == 0.7 and cfg.costs.review_eur == 5.0
    assert len(digest) == 64


def test_the_hash_changes_with_the_file(tmp_path, config_text):
    a = config.load(write(tmp_path, config_text()))[1]
    b = config.load(write(tmp_path, config_text(seed=1)))[1]
    assert a != b


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"url": "http://example.org/x.pq"}, "https"),
        ({"sha256": "abc"}, "sha256"),
        ({"train": 0.8}, "add up"),
        ({"review_eur": 0}, "review_eur"),
        ({"max_bytes": -1}, "max_bytes"),
        ({"seed": -1}, "seed"),
    ],
)
def test_bad_values_are_refused(tmp_path, config_text, overrides, message):
    with pytest.raises(config.ConfigError, match=message):
        config.load(write(tmp_path, config_text(**overrides)))


def test_unknown_keys_are_refused(tmp_path, config_text):
    text = config_text().replace("  review_eur:", "  reveiw_eur: 5.0\n  review_eur:")
    with pytest.raises(config.ConfigError, match="reveiw_eur"):
        config.load(write(tmp_path, text))


@pytest.mark.parametrize("text", ["- just\n- a list\n", "data: [unclosed\n", ""])
def test_unreadable_files_are_refused(tmp_path, text):
    with pytest.raises(config.ConfigError):
        config.load(write(tmp_path, text))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(config.ConfigError, match="can't read"):
        config.load(tmp_path / "nope.yaml")


def test_an_unquoted_all_digit_hash_is_refused_not_misread(tmp_path, config_text):
    text = config_text(sha="1" * 64).replace('"' + "1" * 64 + '"', "1" * 64)
    with pytest.raises(config.ConfigError, match="sha256"):
        config.load(write(tmp_path, text))
