"""
Configuration utilities for reading config.yaml parameters.
"""
import yaml
from pathlib import Path

def load_config(config_file="../Input/config.yaml"):
    """Load configuration from config.yaml file."""
    try:
        config_path = Path(__file__).parent.parent / "Input" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
        return None

def get_ntimesteps(config_file="../Input/config.yaml"):
    """Get number of timesteps from config file."""
    config = load_config(config_file)
    if config and 'General' in config and 'nTimesteps' in config['General']:
        return config['General']['nTimesteps']
    else:
        print("⚠️  Warning: Could not read nTimesteps from config, defaulting to 288")
        return 288

def get_config_value(section, key, default=None, config_file="../Input/config.yaml"):
    """Get a specific configuration value."""
    config = load_config(config_file)
    if config and section in config and key in config[section]:
        return config[section][key]
    else:
        if default is not None:
            print(f"⚠️  Warning: Could not read {section}.{key} from config, using default: {default}")
        return default 