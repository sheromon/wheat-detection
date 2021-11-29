"""Config loading and validation"""
from pathlib import Path

import configobj
from validate import Validator

# TODO: use ResourceManager API as recommended in
# https://setuptools.pypa.io/en/latest/userguide/datafiles.html
CONFIG_DIR = Path(__file__).parent


def load_config(config_path=None, configspec_path=None, do_validation=True):
    """Load config.

    :param str config_path: optional path to configuration file with
        model-specific parameters; if not provided, load default config.ini
    :param str configspec_path: optional path to model configspec file
    :param bool do_validation: optional; if True (default), run validation

    :return: ConfigObj object, basically a dictionary of parameters
    """
    if config_path is None:
        config_path = CONFIG_DIR/'config.ini'
    else:
        config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No file '{config_path}'.")
    # Assumes that the configspec.ini for the given config file is in the same directory
    if not configspec_path:
        configspec_path = config_path.parent/'configspec.ini'

    spec = configobj.ConfigObj(str(configspec_path), list_values=False)
    config = configobj.ConfigObj(str(config_path), configspec=spec)

    # Read config file and assign appropriate types to values
    validator = Validator()

    result = config.validate(validator, preserve_errors=True)
    if do_validation and result is not True:
        for key, value in result.items():
            if value is not True:
                raise RuntimeError(f'Config parameter "{key}": {value}')

    return config
