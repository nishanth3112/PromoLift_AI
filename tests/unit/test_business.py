"""Unit tests for the business config and the average basket value."""

from pathlib import Path

import pytest

from promolift.optimization.business import (
    BUSINESS_CONFIG_PATH,
    average_transaction_value,
    load_business_config,
)

_VALID = """
currency: RUB
sms_cost: 3.0
gross_margin: 0.25
margin_per_purchase: null
budget: null
campaign_clients: null
break_even_grid: [0.01, 0.05]
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "business.yaml"
    path.write_text(text)
    return path


def test_derives_the_margin_from_the_basket_when_not_set(tmp_path: Path) -> None:
    config = load_business_config(_write(tmp_path, _VALID))

    economics = config.economics(average_transaction=400.0)

    assert economics.margin_per_purchase == pytest.approx(100.0)
    assert economics.sms_cost == 3.0
    assert economics.break_even_uplift == pytest.approx(0.03)
    assert config.break_even_grid == (0.01, 0.05)


def test_an_explicit_margin_overrides_the_basket(tmp_path: Path) -> None:
    config = load_business_config(
        _write(tmp_path, _VALID.replace("margin_per_purchase: null", "margin_per_purchase: 60"))
    )

    assert config.margin(average_transaction=400.0) == 60.0


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("sms_cost: 3.0", "sms_cost: -1", ValueError),
        ("gross_margin: 0.25", "gross_margin: 1.5", ValueError),
        ("margin_per_purchase: null", "margin_per_purchase: 0", ValueError),
        ("budget: null", "budget: 1000", ValueError),  # without campaign_clients
        ("break_even_grid: [0.01, 0.05]", "break_even_grid: []", ValueError),
        ("sms_cost: 3.0\n", "", KeyError),
    ],
)
def test_rejects_invalid_configs(tmp_path: Path, old: str, new: str, error: type) -> None:
    with pytest.raises(error):
        load_business_config(_write(tmp_path, _VALID.replace(old, new)))


def test_a_budget_needs_the_clients_it_covers(tmp_path: Path) -> None:
    text = _VALID.replace("budget: null", "budget: 1000").replace(
        "campaign_clients: null", "campaign_clients: 5000"
    )

    config = load_business_config(_write(tmp_path, text))

    assert (config.budget, config.campaign_clients) == (1000.0, 5000)


def test_the_committed_config_is_valid() -> None:
    config = load_business_config(Path(__file__).resolve().parents[2] / BUSINESS_CONFIG_PATH)

    assert config.sms_cost > 0
    assert config.break_even_grid == tuple(sorted(config.break_even_grid))
    assert config.send_share is not None
    assert 0 < config.send_share < 1


def test_send_share_is_optional_and_must_be_a_fraction(tmp_path: Path) -> None:
    assert load_business_config(_write(tmp_path, _VALID)).send_share is None
    assert load_business_config(_write(tmp_path, _VALID + "send_share: 0.38\n")).send_share == 0.38
    with pytest.raises(ValueError, match="send_share"):
        load_business_config(_write(tmp_path, _VALID + "send_share: 1.5\n"))


def test_average_transaction_counts_each_transaction_once(feature_raw_dir: Path) -> None:
    # Transaction t1 has two product rows repeating purchase_sum 100; t2 is 50.
    assert average_transaction_value(feature_raw_dir) == pytest.approx(75.0)
