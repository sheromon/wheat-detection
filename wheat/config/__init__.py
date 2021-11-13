from pathlib import Path

import configobj
from validate import Validator


def load_config(config_path, configspec_path=None, do_validation=True):
    """Load config.

    :param str config_path: path to configuration file with model-specific
        parameters
    :param str configspec_path: optional path to model configspec file
    :param bool do_validation: optional; if True (default), run validation

    :return: ConfigObj object, basically a dictionary of parameters
    """
    # Assumes that the configspec.ini for the given config file is in the same directory
    if not configspec_path:
        configspec_path = Path(config_path).parent/'configspec.ini'

    spec = configobj.ConfigObj(str(configspec_path), list_values=False)
    config = configobj.ConfigObj(str(config_path), configspec=spec)

    # Read config file and assign appropriate types to values
    validator = Validator()

    result = config.validate(validator, preserve_errors=True)
    if do_validation and result is not True:
        for key, value in result.items():
            if value is not True:
                raise RuntimeError('Config parameter "%s": %s' % (key, value))

    return config
