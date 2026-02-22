"""
Global constants for the trading bot.
"""

from enum import Enum
from typing import Final

# Symbol
SYMBOL: Final[str] = "XAUUSDm"

# Timeframes (MetaTrader 5 format)
class Timeframe(Enum):
    """MT5 Timeframe constants."""
    M1 = 1
    M5 = 5
    M15 = 15
    M30 = 30
    H1 = 60
    H4 = 240
    D1 = 1440
    W1 = 10080

    @classmethod
    def from_string(cls, timeframe_str: str) -> "Timeframe":
        """Convert string to Timeframe enum."""
        mapping = {
            "M1": cls.M1,
            "M5": cls.M5,
            "M15": cls.M15,
            "M30": cls.M30,
            "H1": cls.H1,
            "H4": cls.H4,
            "D1": cls.D1,
            "W1": cls.W1,
        }
        return mapping.get(timeframe_str.upper(), cls.M15)

    def to_minutes(self) -> int:
        """Convert timeframe to minutes."""
        return self.value

# Trade Types
class TradeType(Enum):
    """Trade type constants."""
    BUY = "BUY"
    SELL = "SELL"

# Trade States
class TradeState(Enum):
    """Trade state constants."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

# Signal Types
class SignalType(Enum):
    """Signal type constants."""
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NEUTRAL = "NEUTRAL"

# Market Conditions
class MarketCondition(Enum):
    """Market condition types."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"

# Volatility Levels
class VolatilityLevel(Enum):
    """Volatility level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Trend Direction
class TrendDirection(Enum):
    """Trend direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

# Trading Sessions
class TradingSession(Enum):
    """Trading session types."""
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"
    NONE = "NONE"

# SMC Structure Types
class StructureType(Enum):
    """SMC structure types."""
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHoCH"  # Change of Character
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"

# Order Block Types
class OrderBlockType(Enum):
    """Order block types."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"

# Fair Value Gap Types
class FVGType(Enum):
    """Fair Value Gap types."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"

# Liquidity Types
class LiquidityType(Enum):
    """Liquidity level types."""
    HIGH = "HIGH"  # Equal highs
    LOW = "LOW"  # Equal lows
    SWEPT = "SWEPT"  # Liquidity swept

# Market Regime Classification
class MarketRegime(Enum):
    """Market regime for adaptive parameter selection."""
    STRONG_TREND_UP = "STRONG_TREND_UP"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    WEAK_TREND_UP = "WEAK_TREND_UP"
    WEAK_TREND_DOWN = "WEAK_TREND_DOWN"
    RANGE_TIGHT = "RANGE_TIGHT"
    RANGE_WIDE = "RANGE_WIDE"
    VOLATILE_BREAKOUT = "VOLATILE_BREAKOUT"
    REVERSAL = "REVERSAL"

    @property
    def category(self) -> str:
        """Get regime category for weight lookup."""
        if self in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
            return "trending"
        elif self in (MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN):
            return "trending"
        elif self in (MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_WIDE):
            return "ranging"
        elif self == MarketRegime.VOLATILE_BREAKOUT:
            return "breakout"
        elif self == MarketRegime.REVERSAL:
            return "reversal"
        return "trending"

    @property
    def is_trending(self) -> bool:
        return self in (
            MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN,
            MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN,
        )

    @property
    def is_bullish(self) -> bool:
        return self in (MarketRegime.STRONG_TREND_UP, MarketRegime.WEAK_TREND_UP)

    @property
    def is_bearish(self) -> bool:
        return self in (MarketRegime.STRONG_TREND_DOWN, MarketRegime.WEAK_TREND_DOWN)

# Constants
PIPS_FACTOR: Final[float] = 0.01  # For XAUUSDm, 1 pip = 0.01 (10 points on 3-digit broker)
POINT_VALUE: Final[float] = 0.001  # Minimum price change (Exness 3-digit)

# Default values
DEFAULT_BARS: Final[int] = 1000
MAX_BARS: Final[int] = 10000
DEFAULT_LOT_SIZE: Final[float] = 0.01

# File paths
LOG_DIR: Final[str] = "logs"
DATA_DIR: Final[str] = "data"
CONFIG_DIR: Final[str] = "config"

# Time constants (seconds)
ONE_MINUTE: Final[int] = 60
FIVE_MINUTES: Final[int] = 300
FIFTEEN_MINUTES: Final[int] = 900
ONE_HOUR: Final[int] = 3600
ONE_DAY: Final[int] = 86400
