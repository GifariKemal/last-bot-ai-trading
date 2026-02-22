"""
Main entry point for XAUUSD Trading Bot.
Smart Money Concepts (SMC) Strategy with Multi-Timeframe Analysis.
"""

import sys
import signal
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bot.trading_bot import TradingBot
from src.bot_logger import setup_logger, get_logger
from src.utils.config_loader import ConfigLoader


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XAUUSD Gold Trading Bot - Smart Money Concepts Strategy"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "demo", "test"],
        default="demo",
        help="Trading mode: live (real account), demo (demo account), test (dry run)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="settings",
        help="Configuration file name (without .yaml)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip live mode confirmation prompt",
    )

    return parser.parse_args()


def load_configuration(config_name: str):
    """
    Load bot configuration.

    Args:
        config_name: Configuration file name

    Returns:
        Complete configuration dictionary
    """
    config_loader = ConfigLoader()

    # Load all configuration files
    settings = config_loader.load(config_name)
    mt5_config = config_loader.load("mt5_config")
    trading_rules = config_loader.load("trading_rules")
    risk_config = config_loader.load("risk_config")
    session_config = config_loader.load("session_config")

    # Merge configurations — custom settings take priority over trading_rules defaults.
    # Rule: use the custom-config value if present, else fall back to trading_rules.
    # This ensures optimized_config_*.yaml values are NOT silently overwritten by defaults.
    def _prefer(custom_key: str, tr_key: str, fallback=None):
        """Return custom settings[key] if non-empty, else trading_rules[key], else fallback."""
        val = settings.get(custom_key)
        return val if val else trading_rules.get(tr_key, fallback or {})

    config = {
        **settings,
        "mt5": mt5_config,
        "strategy":            _prefer("strategy",            "strategy"),
        "indicators":          _prefer("indicators",          "indicators"),
        "smc_indicators":      _prefer("smc_indicators",      "smc_indicators"),
        "technical_indicators":_prefer("technical_indicators","technical_indicators"),
        "confluence_weights":  _prefer("confluence_weights",  "confluence_weights"),
        "market_conditions":   _prefer("market_conditions",   "market_conditions"),
        "mtf_analysis":        _prefer("mtf_analysis",        "mtf_analysis"),
        "signal_validation":   _prefer("signal_validation",   "signal_validation"),
        "risk":   settings.get("risk")    or risk_config,
        "session":settings.get("session") or session_config,
    }

    return config


def display_startup_banner(mode: str, logger):
    """Display startup banner."""
    logger.info("=" * 80)
    logger.info("XAUUSD GOLD TRADING BOT")
    logger.info("Smart Money Concepts (SMC) Strategy")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Symbol: XAUUSDm")
    logger.info(f"Primary Timeframe: M15")
    logger.info(f"Multi-Timeframe Analysis: H1, M15, M5, M1")
    logger.info("=" * 80)


def display_strategy_info(logger, config: dict):
    """Display strategy information read from live config (no hardcoded backtest stats)."""
    risk   = config.get("risk", {})
    sl_cfg = risk.get("stop_loss", {})
    tp_cfg = risk.get("take_profit", {})
    exits  = risk.get("exit_stages", {})
    pos    = risk.get("position_limits", {})
    strat  = config.get("strategy", {}).get("entry", {})

    be_rr    = exits.get("be_trigger_rr", "?")
    pc_rr    = exits.get("partial_close_rr", "?")
    tr_rr    = exits.get("trail_activation_rr", "?")
    max_pos  = pos.get("max_open_positions", "?")
    min_conf = strat.get("min_confluence_score", "?")

    # Per-regime values from settings.yaml (used when use_adaptive_scorer: true)
    use_adaptive   = config.get("use_adaptive_scorer", False)
    regime_weights = config.get("regime_weights", {})
    regime_conf    = {r: v["min_confluence"] for r, v in regime_weights.items() if "min_confluence" in v}
    regime_sl      = {r: v["atr_sl_mult"] for r, v in regime_weights.items() if "atr_sl_mult" in v}

    logger.info("\nStrategy Components (SMC):")
    logger.info("  * Fair Value Gaps (FVG) - Market imbalances")
    logger.info("  * Order Blocks (OB) - Institutional zones")
    logger.info("  * Liquidity Sweeps - Stop hunts")
    logger.info("  * Break of Structure (BOS) - Continuation")
    logger.info("  * Change of Character (CHoCH) - Reversal")
    logger.info("\nRisk Management (Live Config):")
    if use_adaptive and regime_sl:
        sl_min = min(regime_sl.values())
        sl_max = max(regime_sl.values())
        logger.info(f"  * Stop Loss  - {sl_min:.1f}x–{sl_max:.1f}x ATR (regime-adaptive)")
    else:
        sl_cfg = risk.get("stop_loss", {})
        logger.info(f"  * Stop Loss  - {sl_cfg.get('atr_multiplier', '?')}x ATR")
    tp_cfg = risk.get("take_profit", {})
    logger.info(f"  * Take Profit - {tp_cfg.get('atr_multiplier', '?')}x ATR")
    logger.info(f"  * Multi-Stage Exit: BE at {be_rr}R | Partial at {pc_rr}R | Trail at {tr_rr}R")
    logger.info(f"  * Max Positions: {max_pos} | SMC: {'V4 Library' if config.get('use_smc_v4') else 'V3 Custom'}")
    if use_adaptive and regime_conf:
        logger.info("  * Min Confluence (V3 adaptive per-regime):")
        for regime_name, val in regime_conf.items():
            logger.info(f"      {regime_name}: {val}")
    else:
        logger.info(f"  * Min Confluence: {min_conf} (V2 fixed)")
    logger.info("\nSession Trading (All Sessions Active):")
    logger.info("  * Overlap : 13:00-16:00 UTC | London: 08:00-16:00 UTC")
    logger.info("  * New York: 13:00-22:00 UTC | Asian : 00:00-08:00 UTC")
    logger.info("")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logger()
    logger = get_logger()

    bot = None

    # Handle SIGTERM gracefully (from kill command or process manager)
    def handle_sigterm(signum, frame):
        logger.info("SIGTERM received - shutting down gracefully (positions kept open)")
        if bot:
            bot.stop(close_positions=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        # Display startup information
        display_startup_banner(args.mode, logger)

        # Load configuration
        logger.info("Loading configuration...")
        config = load_configuration(args.config)

        # Display strategy info (reads from live config — no hardcoded backtest stats)
        display_strategy_info(logger, config)

        # Warn if live mode
        if args.mode == "live" and not args.yes:
            logger.warning("=" * 80)
            logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
            logger.warning("=" * 80)
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Live trading cancelled by user")
                sys.exit(0)

        # Initialize and start bot
        logger.info("Initializing trading bot...")
        bot = TradingBot(config)

        logger.info("Starting bot...")
        bot.start()

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("Bot stopped by user (Ctrl+C) - positions kept open")
        logger.info("=" * 80)
        if bot:
            bot.stop(close_positions=False)
        sys.exit(0)

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        # Don't close positions on crash
        if bot:
            bot.stop(close_positions=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
