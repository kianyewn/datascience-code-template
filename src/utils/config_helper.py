from abc import ABC, abstractmethod
from typing import Dict

import yaml
from loguru import logger

from .file_helper import FileHandler, PathParser


class Reader(ABC):
    @abstractmethod
    def load(self, file_location):
        pass


class Writer(ABC):
    @abstractmethod
    def save(self, file_location):
        pass


class ConfigYAML(Reader, Writer):
    @staticmethod
    def load(yaml_path: str):
        with open(yaml_path, "r") as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        logger.info(f"Successfully loaded file from `{yaml_path}`.")
        return config

    @staticmethod
    def save(obj: Dict, yaml_path: str):
        with open(yaml_path, "w") as stream:
            yaml.dump(obj, stream)
        logger.info(f"Successfully saved file to  `{yaml_path}`.")
        return

    @staticmethod
    def delete(yaml_path: str) -> None:
        FileHandler.remove_file(yaml_path)
        logger.info(f"Successfully deleted file from `{yaml_path}`.")


class ConfigClass:
    def __init__(self, config: Dict):
        for key, value in config.items():
            setattr(self, key, value)

    def dict(self):
        return self.__dict__


if __name__ == "__main__":
    mydict = {"a": 1, "b": 2, "c": 3}
    conf = ConfigClass(mydict)
    print(conf.dict())

    ConfigYAML.save(obj=mydict, yaml_path="data/config.yaml")

    import time

    time.sleep(3)

    ConfigYAML.delete(yaml_path="data/config.yaml")
