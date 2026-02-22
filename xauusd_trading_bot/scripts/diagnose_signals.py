"""Quick diagnostic: check what SMC signals exist in current M15 data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mt5_connector import MT5Connector
from src.core.data_manager import DataManager
from src.indicators.technical import TechnicalIndicators
from src.indicators.smc_indicators import SMCIndicators
from src.utils.config_loader import ConfigLoader

config_loader = ConfigLoader()
settings = config_loader.load("settings")
indicators_config = config_loader.load("trading_rules").get("indicators", {})

mt5 = MT5Connector()
mt5.connect()

dm = DataManager()
ti = TechnicalIndicators(indicators_config)
smc = SMCIndicators(indicators_config)

# Fetch M15 data
import MetaTrader5 as mt5lib
df_pandas = mt5.get_bars("XAUUSDm", mt5lib.TIMEFRAME_M15, 1000)
df = dm.pandas_to_polars(df_pandas)
df = dm.add_basic_features(df)
df = dm.add_price_changes(df)
df = ti.calculate_all(df)
df = smc.calculate_all(df)

tick = mt5.get_tick("XAUUSDm")
price = tick["bid"]

print(f"\n{'='*60}")
print(f"XAUUSD M15 Diagnostic | Price: {price:.2f}")
print(f"Data: {len(df)} bars | Last: {df['time'][-1]}")
print(f"{'='*60}")

# Check structure columns
for col in ["is_bullish_bos", "is_bearish_bos", "is_bullish_choch", "is_bearish_choch"]:
    if col in df.columns:
        total = df[col].sum()
        last_50 = df.tail(50)[col].sum()
        last_20 = df.tail(20)[col].sum()
        last_10 = df.tail(10)[col].sum()
        print(f"  {col}: total={total}, last50={last_50}, last20={last_20}, last10={last_10}")
    else:
        print(f"  {col}: COLUMN MISSING!")

# Check FVG columns
print(f"\nFVG Detection:")
for col in ["is_bullish_fvg", "is_bearish_fvg"]:
    if col in df.columns:
        total = df[col].sum()
        last_50 = df.tail(50)[col].sum()
        print(f"  {col}: total={total}, last50={last_50}")

# Check OB columns
print(f"\nOrder Block Detection:")
for col in ["is_bullish_ob", "is_bearish_ob"]:
    if col in df.columns:
        total = df[col].sum()
        last_50 = df.tail(50)[col].sum()
        print(f"  {col}: total={total}, last50={last_50}")

# Check liquidity (actual column names)
print(f"\nLiquidity Detection:")
for col in ["swept_high_liquidity", "swept_low_liquidity"]:
    if col in df.columns:
        total = df[col].sum()
        last_50 = df.tail(50)[col].sum()
        last_20 = df.tail(20)[col].sum()
        print(f"  {col}: total={total}, last50={last_50}, last20={last_20}")
    else:
        print(f"  {col}: COLUMN MISSING!")
# Also check all columns available
liq_cols = [c for c in df.columns if "liq" in c.lower() or "swept" in c.lower()]
print(f"  Available liquidity columns: {liq_cols}")

# Now check what get_bullish/bearish signals return
bull = smc.get_bullish_signals(df, price)
bear = smc.get_bearish_signals(df, price)

print(f"\n{'='*60}")
print(f"LIVE SMC SIGNALS @ {price:.2f}")
print(f"{'='*60}")
print(f"BULLISH: FVG={bull['fvg']['in_zone']}, OB={bull['order_block']['at_zone']}, "
      f"Liq={bull['liquidity']['swept']}, BOS={bull['structure']['bos']}, "
      f"CHoCH={bull['structure']['choch']}, Score={bull['confluence_score']:.2f}")
print(f"BEARISH: FVG={bear['fvg']['in_zone']}, OB={bear['order_block']['at_zone']}, "
      f"Liq={bear['liquidity']['swept']}, BOS={bear['structure']['bos']}, "
      f"CHoCH={bear['structure']['choch']}, Score={bear['confluence_score']:.2f}")

# Check where the last BOS/CHoCH occurred
print(f"\nLast structure breaks in entire dataset:")
for col in ["is_bullish_bos", "is_bearish_bos", "is_bullish_choch", "is_bearish_choch"]:
    if col in df.columns:
        mask = df[col]
        if mask.sum() > 0:
            # Find last True index
            last_idx = len(df) - 1 - mask.reverse().to_list().index(True)
            bars_ago = len(df) - 1 - last_idx
            time_val = df["time"][last_idx]
            print(f"  {col}: last at bar {last_idx} ({bars_ago} bars ago) = {time_val}")
        else:
            print(f"  {col}: NEVER detected in 1000 bars!")

mt5.disconnect()
print(f"\nDone.")
