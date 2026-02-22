"""
Configuration loader utility.
Loads YAML configuration files and provides easy access to settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}

    def load(self, config_name: str, required: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load a configuration file.

        Args:
            config_name: Name of config file (without .yaml extension)
            required: If True, raise error if file not found

        Returns:
            Configuration dictionary or None if not found and not required

        Raises:
            FileNotFoundError: If required config file not found
            yaml.YAMLError: If config file is invalid YAML
        """
        # Check if already loaded
        if config_name in self._configs:
            return self._configs[config_name]

        # Build file path
        config_path = self.config_dir / f"{config_name}.yaml"

        # Check if file exists
        if not config_path.exists():
            if required:
                raise FileNotFoundError(f"Required config file not found: {config_path}")
            return None

        # Load YAML file
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Cache the config
            self._configs[config_name] = config
            return config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")

    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            config_name: Name of config file
            key_path: Dot-separated path to value (e.g., "connection.timeout")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> loader = ConfigLoader()
            >>> timeout = loader.get("mt5_config", "connection.timeout", 60000)
        """
        config = self.load(config_name, required=False)
        if config is None:
            return default

        # Navigate through nested dictionaries
        keys = key_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files in the config directory.

        Returns:
            Dictionary mapping config names to their contents
        """
        if not self.config_dir.exists():
            return {}

        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            if config_name not in self._configs:
                self.load(config_name, required=False)

        return self._configs

    def reload(self, config_name: Optional[str] = None) -> None:
        """
        Reload configuration file(s).

        Args:
            config_name: Specific config to reload, or None to reload all
        """
        if config_name:
            if config_name in self._configs:
                del self._configs[config_name]
            self.load(config_name, required=False)
        else:
            self._configs.clear()
            self.load_all()

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded configurations.

        Returns:
            Dictionary of all loaded configs
        """
        return self._configs.copy()


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """
    Get the global configuration loader instance.

    Args:
        config_dir: Path to configuration directory

    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def load_config(config_name: str, required: bool = True) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load a configuration file.

    Args:
        config_name: Name of config file (without .yaml extension)
        required: If True, raise error if file not found

    Returns:
        Configuration dictionary or None
    """
    loader = get_config_loader()
    return loader.load(config_name, required)


def get_config_value(config_name: str, key_path: str, default: Any = None) -> Any:
    """
    Convenience function to get a configuration value.

    Args:
        config_name: Name of config file
        key_path: Dot-separated path to value
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    loader = get_config_loader()
    return loader.get(config_name, key_path, default)
