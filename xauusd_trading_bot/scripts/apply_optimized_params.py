"""
Apply Optimized Parameters Script
Apply optimized parameters from Optuna study to bot configuration.
"""

import sys
from pathlib import Path
import argparse
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot_logger import setup_logger, get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply optimized parameters to bot configuration"
    )

    parser.add_argument(
        "study_name",
        type=str,
        help="Name of the optimization study",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without applying them",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of current config",
    )

    return parser.parse_args()


def load_optimization_results(study_name: str):
    """Load optimization results."""
    results_dir = Path("data/optimization_results")
    results_file = results_dir / f"{study_name}_results.json"
    config_file = results_dir / f"{study_name}_best_config.yaml"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(results_file, 'r') as f:
        results = json.load(f)

    with open(config_file, 'r') as f:
        best_config = yaml.safe_load(f)

    return results, best_config


def backup_config(config_path: Path, logger):
    """Create backup of config file."""
    if not config_path.exists():
        return

    backup_path = config_path.with_suffix('.yaml.bak')
    backup_path.write_text(config_path.read_text())
    logger.info(f"Backup created: {backup_path}")


def apply_to_trading_rules(best_config: dict, logger, dry_run: bool = False, backup: bool = False):
    """Apply parameters to trading_rules.yaml."""
    config_path = Path("config/trading_rules.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    # Backup if requested
    if backup and not dry_run:
        backup_config(config_path, logger)

    # Load current config
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f)

    # Update strategy section
    if "strategy" in best_config:
        if "strategy" not in current_config:
            current_config["strategy"] = {}
        current_config["strategy"].update(best_config["strategy"])
        logger.info("✓ Updated strategy parameters")

    # Update indicators section
    if "indicators" in best_config:
        if "indicators" not in current_config:
            current_config["indicators"] = {}
        current_config["indicators"].update(best_config["indicators"])
        logger.info("✓ Updated indicator parameters")

    # Update confluence weights
    if "confluence_weights" in best_config:
        current_config["confluence_weights"] = best_config["confluence_weights"]
        logger.info("✓ Updated confluence weights")

    if dry_run:
        logger.info("\n[DRY RUN] Would update trading_rules.yaml:")
        print(yaml.dump(current_config, default_flow_style=False, sort_keys=False))
    else:
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"✓ Applied to {config_path}")

    return True


def apply_to_risk_config(best_config: dict, logger, dry_run: bool = False, backup: bool = False):
    """Apply parameters to risk_config.yaml."""
    config_path = Path("config/risk_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    # Backup if requested
    if backup and not dry_run:
        backup_config(config_path, logger)

    # Load current config
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f)

    # Update risk section
    if "risk" in best_config:
        for key, value in best_config["risk"].items():
            if key in current_config:
                current_config[key].update(value)
                logger.info(f"✓ Updated {key}")
            else:
                current_config[key] = value
                logger.info(f"✓ Added {key}")

    if dry_run:
        logger.info("\n[DRY RUN] Would update risk_config.yaml:")
        print(yaml.dump(current_config, default_flow_style=False, sort_keys=False))
    else:
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"✓ Applied to {config_path}")

    return True


def apply_to_session_config(best_config: dict, logger, dry_run: bool = False, backup: bool = False):
    """Apply parameters to session_config.yaml."""
    config_path = Path("config/session_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    # Backup if requested
    if backup and not dry_run:
        backup_config(config_path, logger)

    # Load current config
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f)

    # Update session section
    if "session" in best_config:
        current_config.update(best_config["session"])
        logger.info("✓ Updated session parameters")

    if dry_run:
        logger.info("\n[DRY RUN] Would update session_config.yaml:")
        print(yaml.dump(current_config, default_flow_style=False, sort_keys=False))
    else:
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"✓ Applied to {config_path}")

    return True


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    setup_logger()
    logger = get_logger()

    try:
        logger.info("=" * 80)
        logger.info("APPLYING OPTIMIZED PARAMETERS")
        logger.info("=" * 80)
        logger.info(f"Study: {args.study_name}")
        if args.dry_run:
            logger.info("Mode: DRY RUN (no changes will be made)")
        if args.backup:
            logger.info("Backup: Enabled")
        logger.info("=" * 80)

        # Load optimization results
        logger.info("\nLoading optimization results...")
        results, best_config = load_optimization_results(args.study_name)

        logger.info(f"Best Score: {results['best_score']:.4f}")
        logger.info(f"Optimization Date: {results['optimization_date']}")

        # Apply to config files
        logger.info("\nApplying parameters to config files...")

        success = True
        success &= apply_to_trading_rules(best_config, logger, args.dry_run, args.backup)
        success &= apply_to_risk_config(best_config, logger, args.dry_run, args.backup)
        success &= apply_to_session_config(best_config, logger, args.dry_run, args.backup)

        if success:
            logger.info("\n" + "=" * 80)
            if args.dry_run:
                logger.info("✓ DRY RUN COMPLETE - Review changes above")
                logger.info("\nTo apply changes, run without --dry-run:")
                logger.info(f"  python scripts/apply_optimized_params.py {args.study_name} --backup")
            else:
                logger.info("✓ PARAMETERS APPLIED SUCCESSFULLY")
                logger.info("\nNext Steps:")
                logger.info("1. Review updated config files in config/")
                logger.info("2. Test with backtest:")
                logger.info("   python scripts/run_backtest.py --months 1")
                logger.info("3. Start demo trading:")
                logger.info("   python main.py --mode demo")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("\n✗ Failed to apply some parameters")
            return 1

    except Exception as e:
        logger.error(f"Error applying parameters: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
