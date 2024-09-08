from configparser import ConfigParser
from typing import Dict


class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, str]:
        """
        Load configuration file that maps original column names to desired column names.
        """
        config = ConfigParser()
        config.read(config_path)
        return {k: v for k, v in config.items("fields")}
