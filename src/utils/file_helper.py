import pathlib
from abc import abstractmethod
from pathlib import Path
from typing import List

from loguru import logger


class FileHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def remove_file(filepath: str) -> None:
        filepath = Path(filepath)
        # Remove the file
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            logger.info(f"Successfully removed file from `{filepath}`.")
        else:
            logger.info("File does not exist.")


class PathParser:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def get_absolute_path(filepath:str) -> str:
        return Path(filepath).resolve()

    @staticmethod
    def get_base_name_from_filepath(filepath: str) -> str:
        return Path(filepath).stem

    @staticmethod
    def get_extensions_from_filepath(filepath: str) -> List:
        return Path(filepath).suffixes

    @staticmethod
    def get_extension_from_filepath(filepath: str) -> str:
        return Path(filepath).suffix

    @staticmethod
    def get_directory_from_filepath(filepath: str) -> pathlib.PosixPath:
        return Path(filepath).parent

    @staticmethod
    def get_path_without_extension(filepath: str) -> pathlib.PosixPath:
        return Path(filepath).with_suffix("")


if __name__ == "__main__":
    filepath = "/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils/file_helper.py"
    pp = PathParser(filepath)
    assert PathParser.get_base_name_from_filepath(filepath) == "file_helper"
    assert pp.get_base_name_from_filepath(filepath) == "file_helper"
    assert pp.get_extension_from_filepath(filepath) == ".py"
    assert pp.get_extensions_from_filepath(filepath) == [
        ".py"
    ], pp.get_extensions_from_filepath(filepath)
    assert pp.get_directory_from_filepath(filepath) == Path(
        "/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils"
    ), pp.get_directory_from_filepath(filepath)

    assert pp.get_path_without_extension(filepath) == Path(
        "/Users/kianyewngieng/github_projects/best_practice_assets/datascience-code-template/src/utils/file_helper"
    )
