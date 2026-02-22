"""
Application configuration.

All tuneable values live here, loaded from a .env file via python-dotenv.
Sub-configs are grouped into dataclasses so callers can do:

    cfg = AppConfig()
    print(cfg.trading.symbol)       # "XAUUSD"
    print(cfg.model.buy_threshold)  # 0.70
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load the .env that sits at the project root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class MT5Config:
    """MetaTrader 5 terminal login credentials."""

    login: int = int(os.getenv("MT5_LOGIN", "0"))
    password: str = os.getenv("MT5_PASSWORD", "")
    server: str = os.getenv("MT5_SERVER", "")
    path: str = os.getenv("MT5_PATH", "")


@dataclass
class TradingConfig:
    """Symbol, timeframe and history-fetch settings."""

    symbol: str = "XAUUSD"
    timeframe: str = "M15"    # must be a key in DataFetcher.TIMEFRAME_MAP
    candles: int = 50_000     # target bars; fetcher batches automatically
    batch_size: int = 10_000  # bars per MT5 request (broker safe-limit)


@dataclass
class ModelConfig:
    """CatBoost training and inference hyper-parameters."""

    test_size: float = 0.20        # fraction of data kept for test evaluation
    random_state: int = 42
    buy_threshold: float = 0.70    # P(BUY) must exceed this to log a fake order
    label_atr_threshold: float = 0.17  # moves < this × ATR are treated as noise and dropped


@dataclass
class AppConfig:
    """Root config that aggregates all sub-configs."""

    mt5: MT5Config = field(default_factory=MT5Config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
