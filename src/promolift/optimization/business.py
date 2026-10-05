"""Campaign economics from ``configs/business.yaml``: SMS cost, margin, optional budget.

Every value is an assumption until the business confirms it, so the config
also lists break-even uplifts (cost / margin) for a sensitivity table: the
best targeting depth depends only on that ratio, so the table answers the
question for any real cost and margin without a re-run.

``margin_per_purchase`` may be left null, meaning: the average transaction
value in ``purchases.csv`` times ``gross_margin``. A converted client is
taken to make one extra transaction of average value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

from promolift.data.loader import Dataset, load_lazy, project_root
from promolift.optimization.targeting import Economics

BUSINESS_CONFIG_PATH = Path("configs") / "business.yaml"


@dataclass(frozen=True)
class BusinessConfig:
    """The campaign economics and the scenarios to report."""

    currency: str
    sms_cost: float
    gross_margin: float
    margin_per_purchase: float | None
    budget: float | None
    campaign_clients: int | None
    break_even_grid: tuple[float, ...]
    send_share: float | None = None

    def margin(self, average_transaction: float) -> float:
        """Margin per extra purchase: the configured value, else basket value x gross margin."""
        if self.margin_per_purchase is not None:
            return self.margin_per_purchase
        return average_transaction * self.gross_margin

    def economics(self, average_transaction: float) -> Economics:
        """The configured SMS cost with the resolved margin."""
        return Economics(self.sms_cost, self.margin(average_transaction))


def _check(config: BusinessConfig) -> None:
    if config.sms_cost < 0:
        raise ValueError("sms_cost must be non-negative")
    if not 0 < config.gross_margin <= 1:
        raise ValueError("gross_margin must be a fraction in (0, 1]")
    if config.margin_per_purchase is not None and config.margin_per_purchase <= 0:
        raise ValueError("margin_per_purchase must be positive (or null to derive it)")
    if (config.budget is None) != (config.campaign_clients is None):
        raise ValueError("budget and campaign_clients must be set together")
    if config.budget is not None and (config.budget < 0 or config.campaign_clients <= 0):
        raise ValueError("budget must be non-negative and campaign_clients positive")
    if not config.break_even_grid or min(config.break_even_grid) < 0:
        raise ValueError("break_even_grid must be a non-empty list of non-negative uplifts")
    if config.send_share is not None and not 0 <= config.send_share <= 1:
        raise ValueError("send_share must be a fraction in [0, 1]")


def load_business_config(path: Path | None = None) -> BusinessConfig:
    """Read and validate the business config.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If a required key is missing.
        ValueError: If a value is out of range.
    """
    path = path if path is not None else project_root() / BUSINESS_CONFIG_PATH
    raw = yaml.safe_load(path.read_text()) or {}
    config = BusinessConfig(
        currency=str(raw["currency"]),
        sms_cost=float(raw["sms_cost"]),
        gross_margin=float(raw["gross_margin"]),
        margin_per_purchase=(
            None if raw.get("margin_per_purchase") is None else float(raw["margin_per_purchase"])
        ),
        budget=None if raw.get("budget") is None else float(raw["budget"]),
        campaign_clients=(
            None if raw.get("campaign_clients") is None else int(raw["campaign_clients"])
        ),
        break_even_grid=tuple(float(r) for r in raw["break_even_grid"]),
        send_share=None if raw.get("send_share") is None else float(raw["send_share"]),
    )
    _check(config)
    return config


def average_transaction_value(base_dir: Path | None = None) -> float:
    """Mean ``purchase_sum`` per transaction over the whole purchase history.

    ``purchases.csv`` has one row per product, repeating the transaction's
    ``purchase_sum``, so rows are de-duplicated per transaction first (as the
    purchase-behavior features do). Streams the file lazily.
    """
    return float(
        load_lazy(Dataset.PURCHASES, base_dir)
        .select("client_id", "transaction_id", "purchase_sum")
        .unique(subset=["client_id", "transaction_id"])
        .select(pl.col("purchase_sum").mean())
        .collect()
        .item()
    )
